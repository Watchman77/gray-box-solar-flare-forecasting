"""Bounded read-only cache inventory; file presence is not content verification.

Run the frozen source with a supplied CONFIG dictionary. This module creates no
VM files, imports no model libraries, downloads no objects, and emits a snapshot
tar stream. A later inference reader must check every selected object's bytes.
"""
import base64
from collections import Counter, defaultdict
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(path, basename, roots):
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts and p.name == basename,
            'Invalid image candidate path')
    require(any(p.is_relative_to(root) for root in roots), 'Candidate outside allowed roots')
    return p


def probe(path, expected_bytes, roots):
    """Never follow links outside the declared existing cache roots."""
    p = safe_path(str(path), path.name, roots)
    try:
        resolved = p.resolve(strict=True)
        require(any(resolved.is_relative_to(root) for root in roots), 'Resolved candidate outside allowed roots')
        st = p.lstat()
        if not stat.S_ISREG(st.st_mode):
            return {'path':str(p), 'status':'nonregular_or_symlink'}
        return {'path':str(p), 'status':'size_match_requires_hash' if st.st_size == expected_bytes else 'size_mismatch',
                'bytes':st.st_size, 'device':st.st_dev, 'inode':st.st_ino,
                'mtime_ns':st.st_mtime_ns, 'ctime_ns':st.st_ctime_ns}
    except FileNotFoundError:
        return {'path':str(p), 'status':'missing'}
    except PermissionError:
        return {'path':str(p), 'status':'unreadable'}


def inventory(rows, roots, scan_root, guard, max_entries=1500000):
    require(len({r['uri'] for r in rows}) == len(rows), 'Duplicate object URI')
    names = {Path(r['uri']).name for r in rows}
    paths = defaultdict(set)
    for row in rows:
        basename = Path(row['uri']).name
        for key in ['candidate_paths_json', 'expected_locations_not_presence_evidence_json']:
            for value in json.loads(row[key]):
                paths[basename].add(str(safe_path(value, basename, roots)))
    scanned = 0
    def walk_error(error):
        raise error
    for directory, dirs, files in os.walk(scan_root, followlinks=False, onerror=walk_error):
        guard()
        dirs[:] = sorted(d for d in dirs if not Path(directory, d).is_symlink())
        scanned += len(files) + len(dirs)
        require(scanned <= max_entries, 'Cache traversal entry bound exceeded')
        for name in files:
            if name in names:
                paths[name].add(str(safe_path(str(Path(directory, name)), name, roots)))
    result = []
    for index, row in enumerate(rows):
        if index % 512 == 0:
            guard()
        candidates = [probe(Path(p), int(row['bytes']), roots) for p in sorted(paths[Path(row['uri']).name])]
        matched = [p for p in candidates if p['status'] == 'size_match_requires_hash']
        result.append({**{k:row[k] for k in ['uri','generation','bytes','md5_base64']},
                       'status':'candidate_requires_content_verification' if matched else 'no_size_matching_candidate',
                       'candidates':candidates, 'image_bytes_verified':False})
    return result, scanned


def main(c):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden')
    require(sys.executable == c['python'], 'Exact VM Python required')
    started = time.monotonic()
    deadline = started + c['maximum_seconds']
    def interrupted(number, frame):
        raise InterruptedError('Cache reconciliation interrupted: ' + str(number))
    for sig in [signal.SIGALRM, signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, c['maximum_seconds'])
    raw = base64.b64decode(c['plan_gzip_base64'], validate=True)
    require(hashlib.sha256(raw).hexdigest() == c['plan_sha256'], 'Cache plan changed')
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(raw).decode())))
    require(len(rows) == 113037, 'Original object support changed')
    lock = Path(c['common_lock'])
    with lock.open('r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def guard():
            require(time.monotonic() < deadline, 'Reconciliation deadline reached')
            require(os.fstat(lease.fileno()).st_ino == lock.stat().st_ino == c['common_lock_inode'], 'Lock identity changed')
            require(sha(c['owner']) == c['owner_sha256'], 'Resource owner changed')
            require(all(shutil.disk_usage(p).free >= c['minimum_free_bytes'] for p in c['disk_roots']), 'Disk reserve violated')
        guard()
        require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                             text=True, timeout=8).strip(), 'GPU occupied')
        records, scanned = inventory(rows, [Path(p) for p in c['allowed_roots']], Path(c['scan_root']),
                                     guard, c['maximum_entries'])
        guard()
        payload = ('\n'.join(json.dumps(r, separators=(',',':')) for r in records)+'\n').encode()
        summary = {'status':'live_path_and_size_reconciliation_complete_not_content_verification',
                   'objects':len(records), 'counts':dict(Counter(r['status'] for r in records)),
                   'no_size_matching_candidate_bytes':sum(int(r['bytes']) for r in records if r['status']=='no_size_matching_candidate'),
                   'scanned_entries':scanned, 'elapsed_seconds':time.monotonic()-started,
                   'plan_sha256':c['plan_sha256'], 'source_sha256':c['source_sha256'],
                   'owner_sha256':c['owner_sha256'], 'common_lock_inode':c['common_lock_inode'],
                   'VM_files_written':0, 'image_downloads':0, 'image_bytes_read':0, 'model_runs':0,
                   'Torch_imported':'torch' in sys.modules, 'GPU_priority_returned':False,
                   'records_sha256':hashlib.sha256(payload).hexdigest(),
                   'records_bytes':len(payload), 'scientific_acceptance':False,
                   'disk_free_bytes':{p:shutil.disk_usage(p).free for p in c['disk_roots']}}
        require(len(payload) < 150*1024**2, 'Inventory output exceeds bound')
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
            for name, content in [('cache_records.jsonl',payload),('summary.json',(json.dumps(summary,indent=2)+'\n').encode())]:
                guard()
                member=tarfile.TarInfo(name); member.size=len(content)
                archive.addfile(member,io.BytesIO(content))
        guard()
    signal.setitimer(signal.ITIMER_REAL,0)
