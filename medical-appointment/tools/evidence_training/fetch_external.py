"""Fetch public annotations and text only; no model, audio, or paid API calls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from .data import write_json

ACI_REVISION = 'b909b2bb9cf1d19de08df15cddde7bd0179665e4'
PRIMOCK_REVISION = 'cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5'


def sources():
    result = {
        'coqa-train-v1.0.json': 'https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json',
        'coqa-dev-v1.0.json': 'https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json',
        'coqa-source.html': 'https://stanfordnlp.github.io/coqa/',
        'simord-train.json': 'https://huggingface.co/datasets/microsoft/SIMORD/resolve/main/data/train.json',
        'simord-dev.json': 'https://huggingface.co/datasets/microsoft/SIMORD/resolve/main/data/dev.json',
        'simord-README.md': 'https://huggingface.co/datasets/microsoft/SIMORD/resolve/main/README.md',
        'simord-builder.py.txt': 'https://huggingface.co/datasets/microsoft/SIMORD/resolve/main/SIMORD.py',
        'mediqa-process.py.txt': 'https://raw.githubusercontent.com/jpcorb20/mediqa-oe/main/data/process_data.py',
        'punkt_tab.zip': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip',
        'CC-BY-SA-4.0.txt': 'https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt',
        'CDLA-Permissive-2.0.html': 'https://cdla.dev/permissive-2-0/',
        'mashqa_data.zip': 'https://drive.google.com/uc?export=download&id=1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK',
        'mashqa-README.md': 'https://raw.githubusercontent.com/mingzhu0527/MASHQA/main/README.md',
        'mashqa-code-LICENSE': 'https://raw.githubusercontent.com/mingzhu0527/MASHQA/main/LICENSE',
    }
    aci = f'https://raw.githubusercontent.com/wyim/aci-bench/{ACI_REVISION}'
    for name in ('train', 'valid', 'clinicalnlp_taskB_test1', 'clinicalnlp_taskC_test2', 'clef_taskC_test3'):
        result[f'aci-{name}.json'] = f'{aci}/data/challenge_data_json/{name}.json'
    result['aci-README.md'] = f'{aci}/README.md'
    result['primock-LICENSE.md'] = (f'https://raw.githubusercontent.com/babylonhealth/primock57/'
                                   f'{PRIMOCK_REVISION}/LICENSE.md')
    return result


def fetch(root, filename, url, expected=None):
    path = root / filename
    if not path.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'evidence-training-data-preparation/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    data = response.read()
                if filename == 'mashqa_data.zip' and not data.startswith(b'PK'):
                    # Public Drive files present a size-warning form; no login is bypassed.
                    page = data.decode()
                    fields = dict(re.findall(r'type="hidden" name="([^"]+)" value="([^"]+)"', page))
                    match = re.search(r'<form id="download-form" action="([^"]+)"', page)
                    if not match or match.group(1) != 'https://drive.usercontent.google.com/download':
                        raise ValueError('Public MASH-QA download unavailable; no credentials supplied')
                    if fields.get('id') != '1RY_gWB4gaUPkW3w9WhIZAwxg5dzNFliK':
                        raise ValueError('Unexpected download target')
                    with urllib.request.urlopen(match.group(1) + '?' + urllib.parse.urlencode(fields), timeout=60) as r:
                        data = r.read()
                    if not data.startswith(b'PK'):
                        raise ValueError('MASH-QA response is not an archive')
                if expected and hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f'Source changed: {filename}')
                if filename.endswith('.json'):
                    json.loads(data)
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(data)
                temporary.replace(path)
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    data = path.read_bytes()
    checksum = hashlib.sha256(data).hexdigest()
    if expected and checksum != expected:
        raise ValueError(f'Cached source differs from lock: {filename}')
    return {'file': filename, 'url': url, 'bytes': len(data), 'sha256': checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, help='Reproduce an earlier download-manifest.json exactly')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'download-manifest.json'
    lock = json.loads(args.lock.read_text()) if args.lock else (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else [])
    expected = {x['file']: x for x in lock}
    urls = {x['file']: x['url'] for x in lock} if args.lock else sources()

    def get(item):
        name, url = item
        entry = fetch(args.output, name, url, expected.get(name, {}).get('sha256'))
        print(f'{name}: {entry["bytes"]} bytes', flush=True)
        return entry

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(get, urls.items()))
    if not args.lock:
        primock_ids = {row['id'] for split in ('train', 'dev')
                       for row in json.loads((args.output / f'simord-{split}.json').read_text())
                       if row['id'].startswith('primock57_')}
        additional = {}
        for cid in sorted(primock_ids):
            _, day, number = cid.split('_')
            for speaker in ('doctor', 'patient'):
                filename = f'day{day}_consultation{int(number):02d}_{speaker}.TextGrid'
                additional[filename] = (f'https://raw.githubusercontent.com/babylonhealth/primock57/'
                                        f'{PRIMOCK_REVISION}/transcripts/{filename}')
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows.extend(pool.map(get, additional.items()))
    write_json(manifest_path, sorted(rows, key=lambda row: row['file']))
    print(f'Saved {len(rows)} sources with SHA-256 checksums to {manifest_path}')


if __name__ == '__main__':
    main()
