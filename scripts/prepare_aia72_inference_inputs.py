"""Pin inference metadata and bounded blocks locally; never download image pixels.

Earlier training object generations remain immutable. New references are pinned
from the user's existing bucket and retain unverified historical availability.
This preparation is not a cache audit, launch allowance or deletion permission.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from scripts.aia72_data import LAGS, validate_case_times
from scripts.aia72_replay_contract import require, sha, write_json

BUCKET = 'suryabench-sharp-pipeline-bamidele'
PARENT = '3d8ad4882433a5670b832d763cb428031b8f18f2deefe4083f03205ce19c87cd'
OLD_OBJECTS = 'ba5b2c12dfac49a7ce78d3b168ba98eb364b99a00b4818798eb4204c4d1aa515'


def parse_uri(uri):
    prefix = 'gs://'+BUCKET+'/'
    require(isinstance(uri, str) and uri.startswith(prefix), 'Unexpected input bucket')
    name = uri[len(prefix):]
    require(name.endswith('.npz') and '..' not in name.split('/') and not name.startswith('/'), 'Invalid object name')
    require(name.startswith(('samples_npz/', 'jsoc_2025_2026_production_v1/samples_npz/')), 'Unexpected image prefix')
    return name


def validate_object(record):
    parse_uri(record['uri'])
    require(isinstance(record['generation'], str) and record['generation'].isdigit(), 'Immutable generation required')
    require(type(record['bytes']) is int and 0 < record['bytes'] <= 16*1024**2, 'Object size outside declared bound')
    require(len(base64.b64decode(record['md5_base64'], validate=True)) == 16, 'Invalid source MD5')


def make_blocks(frame, objects, max_cases=512, max_bytes=14*1024**3, batch_size=16):
    require(frame.forecast_case_id.is_unique, 'Duplicate requested case')
    require(max_cases >= batch_size and max_cases % batch_size == 0, 'Block size must preserve batch boundaries')
    columns = [f'history_uri_tminus{x}' for x in LAGS]
    blocks = []
    for role, subset in frame.groupby('role', sort=False):
        subset = subset.reset_index(drop=True)
        start = 0
        while start < len(subset):
            count = min(max_cases, len(subset)-start)
            while count:
                selected = subset.iloc[start:start+count]
                uris = sorted(set(selected[columns].to_numpy().ravel()))
                require(all(uri in objects for uri in uris), 'Object pin missing; no case dropping')
                total = sum(objects[uri]['bytes'] for uri in uris)
                if total <= max_bytes:
                    break
                count = ((count-1)//batch_size)*batch_size
            require(count > 0, 'One original batch exceeds scratch cap')
            stop = start+count
            require(stop == len(subset) or count % batch_size == 0, 'Nonfinal block changed batching')
            blocks.append({'block_id': f'{role}_{start:06d}_{stop:06d}', 'role': role,
                           'start': start, 'stop': stop, 'case_count': count,
                           'case_ids': selected.forecast_case_id.tolist(), 'uris': uris,
                           'all_images_bytes_upper_bound': total,
                           'image_count': len(uris), 'final_within_role': stop == len(subset)})
            start = stop
    expected = [case for _, group in frame.groupby('role', sort=False) for case in group.forecast_case_id]
    require([case for b in blocks for case in b['case_ids']] == expected, 'Block support/order changed')
    return blocks


def prepare(root, output, gcloud, max_requests=300, timeout_seconds=600):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Preserve every metadata preparation attempt')
    parent = root/'outputs/multimodal72_preparation_v1_20261002T165758076710Z'
    require(sha(parent/'experiment_contract.json') == PARENT, 'Frozen comparison contract changed')
    contract = json.loads((parent/'experiment_contract.json').read_text())
    cases_path = parent/'candidate_comparison_cases.csv.gz'
    require(sha(cases_path) == contract['split_manifest_sha256'], 'Frozen comparison cases changed')
    frame = pd.read_csv(cases_path, low_memory=False)
    require(len(frame) == 96596 and frame.forecast_case_id.is_unique, 'Comparison support changed')
    validate_case_times(frame)
    wanted = set(frame[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
    require(len(wanted) == 113037, 'Image support changed')
    for uri in wanted:
        parse_uri(uri)
    old_path = root/'outputs/aia72_training_inputs_v1_20261002/objects.json'
    require(sha(old_path) == OLD_OBJECTS, 'Training image pins changed')
    old = json.loads(old_path.read_text())
    require(set(old) <= wanted, 'Earlier training pins outside comparison')
    records = dict(old)
    for record in records.values():
        validate_object(record)
    missing = wanted-set(old)
    prefixes = sorted({parse_uri(uri).rsplit('/', 1)[0]+'/' for uri in missing})
    output.mkdir(parents=True)
    pages = output/'metadata_pages'; pages.mkdir()
    started = time.monotonic()
    receipt = {'status': 'metadata_preparation_started', 'image_downloads': 0,
               'GPU_launches': 0, 'VM_operations': 0, 'cloud_mutations': 0,
               'maximum_metadata_requests': max_requests, 'maximum_seconds': timeout_seconds}
    write_json(output/'receipt.json', receipt)
    requests = 0
    try:
        token = subprocess.run([str(gcloud),'auth','print-access-token'], capture_output=True,
                               check=True, text=True, timeout=60).stdout.strip()
        require(token, 'No cloud authentication token')
        for prefix in prefixes:
            page = None
            while True:
                require(requests < max_requests and time.monotonic()-started < timeout_seconds, 'Metadata bound reached')
                params = {'prefix': prefix, 'maxResults': 1000,
                          'fields': 'items(name,generation,size,md5Hash),nextPageToken'}
                if page:
                    params['pageToken'] = page
                url = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?'+urlencode(params)
                with urlopen(Request(url, headers={'Authorization':'Bearer '+token}), timeout=20) as response:
                    raw = response.read(2*1024**2+1)
                require(len(raw) <= 2*1024**2, 'Metadata page exceeds bound')
                requests += 1
                (pages/f'{requests:04d}.json').write_bytes(raw)
                data = json.loads(raw)
                for item in data.get('items', []):
                    uri = f"gs://{BUCKET}/{item['name']}"
                    if uri in missing:
                        record = {'uri':uri,'generation':str(item['generation']),
                                  'bytes':int(item['size']),'md5_base64':item['md5Hash']}
                        validate_object(record)
                        require(uri not in records or records[uri] == record, 'Conflicting object metadata')
                        records[uri] = record
                page = data.get('nextPageToken')
                if not page:
                    break
            print(json.dumps({'completed_prefix':prefix,'pinned_objects':len(records),'of':len(wanted)}), flush=True)
        require(set(records) == wanted, f'Missing required objects: {len(wanted-set(records))}; no silent case dropping')
        require({uri:records[uri] for uri in old} == old, 'Earlier immutable generations changed')
        blocks = make_blocks(frame, records)
        write_json(output/'objects.json', records)
        write_json(output/'blocks.json', blocks)
        (output/'cases.csv.gz').write_bytes(cases_path.read_bytes())
        role_summary = []
        for role, selected in frame.groupby('role', sort=False):
            parts = [b for b in blocks if b['role'] == role]
            role_summary.append({'role':role,'cases':len(selected),'blocks':len(parts),
                                 'largest_block_image_bytes':max(b['all_images_bytes_upper_bound'] for b in parts)})
        receipt.update(status='metadata_and_block_plan_complete_not_inference',
            utc=datetime.now(timezone.utc).isoformat(), metadata_requests=requests,
            elapsed_seconds=time.monotonic()-started, cases=len(frame), objects=len(records),
            reused_training_object_pins=len(old), new_object_pins=len(missing),
            total_unique_image_bytes=sum(r['bytes'] for r in records.values()),
            parent_contract_sha256=PARENT, original_training_objects_sha256=OLD_OBJECTS,
            objects_sha256=sha(output/'objects.json'), cases_sha256=sha(output/'cases.csv.gz'),
            blocks_sha256=sha(output/'blocks.json'), block_count=len(blocks),
            max_block_cases=512, batch_size=16, scratch_cap_bytes=14*1024**3,
            minimum_free_disk_bytes=20*1024**3, role_summary=role_summary,
            historical_availability='unverified_retrospective', scientific_acceptance=False,
            cache_join_performed=False, cache_pixels_verified=False,
            next='Join these exact URI/generation/checksum pins with preserved VM cache inventories; verify actual bytes on use. Freeze reviewed inference/controller bounds after saved-model replay. This receipt authorizes no download, launch or retirement.')
        write_json(output/'receipt.json', receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status='metadata_preparation_failed', error=type(exc).__name__+': '+str(exc),
                       metadata_requests=requests, elapsed_seconds=time.monotonic()-started,
                       automatic_retry=False)
        write_json(output/'receipt.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gcloud', type=Path, default=Path('/Users/mac/google-cloud-sdk/bin/gcloud'))
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.output, args.gcloud), indent=2))
