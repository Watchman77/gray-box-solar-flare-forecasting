"""Join preserved cache hints to exact Gray input pins without contacting the VM."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from scripts.aia72_replay_contract import require, sha, write_json
from scripts.aia72_data import LAGS

INPUTS = {
    'lineage_paths': ('inputs/late_fusion_feasibility_20261001/aia_lineage/cycle24_temporal_targets_local_paths.csv.gz',
                     '8089534872ea158357d552bf58fa1742b37d16122e7d775ae088b04b28affbd2'),
    'lineage_uris': ('inputs/late_fusion_feasibility_20261001/lineage/cycle24_temporal_target_to_object_map.csv.gz',
                    '339b82d629d2b73715ada17b5c0a39b651a2d36c5de246f91d67b1f8d4a0f105'),
    'later_candidates': ('run_records/GRAYBOX_CACHE_METADATA_SHARED_20261004/cache_candidates.csv.gz',
                         'dec0f2266bbb0e87c6164e29cae8de52d882946ad9fe65ae848c83772a8783b9')}


def safe_path(path, uri):
    """Validate path text only; do not resolve/stat/open any VM image path."""
    path = str(path)
    p = PurePosixPath(path)
    require(p.is_absolute() and '..' not in p.parts and p.name == PurePosixPath(uri).name,
            'Unexpected candidate path/identity')
    require(path.startswith(('/mnt/disks/aia-cache/', '/home/abmoses2000/')), 'Candidate outside known VM storage')
    return path


def metadata_matches(pin, generation, size, md5):
    return str(generation) == pin['generation'] and int(size) == pin['bytes'] and str(md5) == pin['md5_base64']


def join_candidates(pins, training_bound, lineage_hints, later):
    require(later.uri.is_unique, 'Duplicate later cache URI')
    later_map = later.set_index('uri').to_dict(orient='index')
    rows = []
    for uri, pin in sorted(pins.items()):
        paths, matched, expected = set(), set(), set()
        sources, mismatches = set(), set()
        for path in lineage_hints.get(uri, []):
            paths.add(safe_path(path, uri)); sources.add('cycle24_lineage_path_hint')
        if uri in training_bound:
            record = training_bound[uri]
            path = safe_path(record['path'], uri)
            if metadata_matches(pin, record['generation'], record['bytes'], record['md5_base64']):
                paths.add(path); matched.add(path); sources.add('gray_training_bound_source')
            else:
                mismatches.add('gray_training_bound_source')
        historical_missing = False
        if uri in later_map:
            record = later_map[uri]
            require(record['status'] in ['MISSING', 'SIZE_MATCH_REQUIRES_CONTENT_HASH'], 'Unexpected historical cache status')
            expected.add(safe_path(record['local_path'], uri))
            agrees = metadata_matches(pin, record['generation'], record['size'], record['md5'])
            if not agrees:
                mismatches.add('aia_later_cache_metadata')
            historical_missing = record['status'] == 'MISSING'
            if historical_missing:
                require(not record['candidate_path'] and int(record['candidate_count']) == 0, 'Missing/candidate fields disagree')
            else:
                require(bool(record['candidate_path']) and int(record['candidate_count']) > 0, 'Historical candidate absent')
                path = safe_path(record['candidate_path'], uri)
                if agrees:
                    paths.add(path); matched.add(path); sources.add('aia_size_match_metadata_candidate')
        category = ('historical_matching_metadata_candidate' if matched else
                    'historical_unversioned_path_hint' if paths else 'no_historical_candidate')
        rows.append({'uri':uri, 'generation':pin['generation'], 'bytes':pin['bytes'],
                     'md5_base64':pin['md5_base64'], 'candidate_category':category,
                     'candidate_paths_json':json.dumps(sorted(paths)),
                     'matching_metadata_paths_json':json.dumps(sorted(matched)),
                     'expected_locations_not_presence_evidence_json':json.dumps(sorted(expected)),
                     'historical_evidence_sources_json':json.dumps(sorted(sources)),
                     'metadata_mismatch_sources_json':json.dumps(sorted(mismatches)),
                     'historically_reported_missing':historical_missing,
                     'live_presence_verified':False, 'image_bytes_verified_for_inference':False})
    return pd.DataFrame(rows)


def prepare(root, aia_root, metadata, output):
    root, aia_root, metadata, output = [Path(p).resolve() for p in (root,aia_root,metadata,output)]
    require(not output.exists(), 'Preserve each cache-planning attempt')
    receipt = json.loads((metadata/'receipt.json').read_text())
    require(receipt['status'] == 'metadata_and_block_plan_complete_not_inference', 'Metadata not completed')
    for name,key in [('objects.json','objects_sha256'),('cases.csv.gz','cases_sha256'),('blocks.json','blocks_sha256')]:
        require(sha(metadata/name) == receipt[key], 'Gray metadata changed: '+name)
    paths = {}
    for name,(relative,digest) in INPUTS.items():
        paths[name] = aia_root/relative
        require(sha(paths[name]) == digest, 'Preserved AIA evidence changed: '+name)
    columns = ['target_sample_id', *[f'history_uri_tminus{x}' for x in LAGS]]
    # Do not load or transfer upstream 48h labels, roles or region assignments.
    lineage = pd.read_csv(paths['lineage_paths'], usecols=columns+[f'local_tminus{x}' for x in LAGS],
                          dtype=str, keep_default_na=False)
    mapping = pd.read_csv(paths['lineage_uris'], usecols=columns, dtype=str, keep_default_na=False)
    require(lineage[columns].equals(mapping[columns]), 'Preserved lineage identities/URIs differ')
    hints = defaultdict(set)
    for lag in LAGS:
        for uri,path in lineage[[f'history_uri_tminus{lag}',f'local_tminus{lag}']].itertuples(index=False,name=None):
            hints[uri].add(path)
    later = pd.read_csv(paths['later_candidates'], dtype=str, keep_default_na=False)
    bound_path = root/'outputs/compute_limit_archive_20261003/snapshot/outputs/training/bound_sources.json'
    require(sha(bound_path) == 'bb02fc64516ac6c5943a3a9737edad88993109a2cd9eafdc6fb9ad88b5ada46a', 'Training cache bindings changed')
    pins = json.loads((metadata/'objects.json').read_text())
    planned = join_candidates(pins,json.loads(bound_path.read_text()),hints,later)
    require(planned.uri.is_unique and set(planned.uri) == set(pins), 'Gray object loss/duplication')
    frame = pd.read_csv(metadata/'cases.csv.gz', low_memory=False)
    index = planned.set_index('uri')
    roles = []
    for role,group in frame.groupby('role',sort=False):
        uris = set(group[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
        subset = index.loc[sorted(uris)]
        counts = subset.candidate_category.value_counts().to_dict()
        roles.append({'role':role,'cases':len(group),'distinct_images':len(uris),
                      'historical_candidate_categories':counts,
                      'no_historical_candidate_bytes':int(subset.loc[subset.candidate_category.eq('no_historical_candidate'),'bytes'].sum())})
    output.mkdir(parents=True)
    planned.to_csv(output/'cache_reuse_plan.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    no_candidate = planned.candidate_category.eq('no_historical_candidate')
    summary = {'status':'offline_historical_cache_join_complete_not_live_verification',
        'utc':datetime.now(timezone.utc).isoformat(),'objects':len(planned),'cases':len(frame),
        'category_counts':planned.candidate_category.value_counts().to_dict(),
        'historically_reported_missing':int(planned.historically_reported_missing.sum()),
        'objects_with_metadata_mismatch':int(planned.metadata_mismatch_sources_json.ne('[]').sum()),
        'no_historical_candidate_bytes':int(planned.loc[no_candidate,'bytes'].sum()),
        'role_summary':roles,'gray_metadata_receipt_sha256':sha(metadata/'receipt.json'),
        'source_code_sha256':sha(Path(__file__)),
        'input_evidence_sha256':{key:sha(path) for key,path in paths.items()},
        'training_bound_sources_sha256':sha(bound_path),'plan_sha256':sha(output/'cache_reuse_plan.csv.gz'),
        'upstream_48h_labels_or_roles_imported':False,'VM_operations':0,'image_downloads':0,
        'live_cache_presence_established':False,'scientific_acceptance':False,
        'limitations':'Historical paths and matching metadata are candidates only. Live file presence and exact content must be checked before reuse. No-candidate bytes estimate planning exposure, not a verified download requirement. Preserve pre-existing files; restoration destinations require a separate owned scratch plan.'}
    write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aia-root',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1],a.aia_root,a.metadata,a.output),indent=2))
