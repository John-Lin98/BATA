"""Download only a SHA-bound GB1 or TrpB reference FASTA, without alteration."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=['GB1', 'TrpB'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads((Path(__file__).resolve().parents[1] / 'configs/REFERENCE_SOURCES.json').read_text())[args.benchmark]
    with urllib.request.urlopen(recipe['url'], timeout=45) as response:
        content = response.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024 or hashlib.sha256(content).hexdigest() != recipe['sha256']:
        raise ValueError('Downloaded reference does not match frozen SHA')
    with args.output.open('xb') as stream:
        stream.write(content)
    print('PASS: exact frozen reference FASTA')


if __name__ == '__main__':
    main()
