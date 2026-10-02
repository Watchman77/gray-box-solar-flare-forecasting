"""Freeze training/selection cases and pin existing cloud objects, without downloading images."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from scripts.dataset_io import file_sha256
from scripts.aia72_data import LAGS, validate_case_times


def prepare(root, output):
    root, output = Path(root), Path(output)
    parent = root / 'outputs/multimodal72_preparation_v1_20261002T165758076710Z'
    contract = json.loads((parent / 'experiment_contract.json').read_text())
    path = parent / 'candidate_comparison_cases.csv.gz'
    if file_sha256(path) != contract['split_manifest_sha256']:
        raise ValueError('Changed frozen case manifest')
    if output.exists():
        raise ValueError('Preserve existing preparation')
    frame = pd.read_csv(path, low_memory=False)
    fit = frame[frame.role.isin(['train', 'model_validation'])].copy()
    validate_case_times(fit)
    if fit.role.value_counts().to_dict() != {'train': 25586, 'model_validation': 3905}:
        raise ValueError('Training/selection support changed')
    wanted = set(fit[[f'history_uri_tminus{x}' for x in LAGS]].to_numpy().ravel())
    token = subprocess.run(['/Users/mac/google-cloud-sdk/bin/gcloud', 'auth', 'print-access-token'],
                           capture_output=True, check=True, text=True, timeout=60).stdout.strip()
    bucket = 'suryabench-sharp-pipeline-bamidele'
    records, pages = {}, 0
    for year in range(2010, 2015):
        page = None
        while True:
            params = {'prefix': f'samples_npz/{year}/', 'maxResults': 1000,
                      'fields': 'items(name,generation,size,md5Hash),nextPageToken'}
            if page:
                params['pageToken'] = page
            pages += 1
            if pages > 100:
                raise ValueError('Metadata request cap exceeded')
            url = f'https://storage.googleapis.com/storage/v1/b/{bucket}/o?' + urlencode(params)
            with urlopen(Request(url, headers={'Authorization': 'Bearer ' + token}), timeout=60) as response:
                data = json.loads(response.read(2 * 1024**2))
            for obj in data.get('items', []):
                uri = f"gs://{bucket}/{obj['name']}"
                if uri in wanted:
                    if not obj.get('md5Hash') or not 0 < int(obj['size']) <= 16 * 1024**2:
                        raise ValueError('Unbounded object or missing checksum')
                    records[uri] = {'uri': uri, 'generation': str(obj['generation']),
                                    'bytes': int(obj['size']), 'md5_base64': obj['md5Hash']}
            page = data.get('nextPageToken')
            if not page:
                break
    if set(records) != wanted:
        raise ValueError(f'Missing {len(wanted - set(records))} required objects; no case dropping')
    output.mkdir(parents=True)
    fit.to_csv(output / 'fit_cases.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    (output / 'objects.json').write_text(json.dumps(records, indent=2) + '\n')
    (output / 'parent_contract.json').write_bytes((parent / 'experiment_contract.json').read_bytes())
    receipt = {'status': 'metadata_pinned_no_images_downloaded', 'utc': datetime.now(timezone.utc).isoformat(),
               'parent_contract_sha256': file_sha256(parent / 'experiment_contract.json'),
               'fit_cases_sha256': file_sha256(output / 'fit_cases.csv.gz'),
               'objects_sha256': file_sha256(output / 'objects.json'),
               'training_cases': 25586, 'selection_cases': 3905, 'images': len(records),
               'metadata_requests': pages, 'image_downloads': 0,
               'source_bytes': sum(r['bytes'] for r in records.values())}
    (output / 'preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.output)
