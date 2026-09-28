"""Validate a separately obtained upstream checkout; never downloads or installs."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkout', type=Path)
args = parser.parse_args()
manifest = json.loads((Path(__file__).resolve().parents[1] /
    'manifests/REAP_DEPENDENCY.json').read_text())
for relative, expected in manifest['files_sha256'].items():
    actual = hashlib.sha256((args.checkout / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('FAIL: source mismatch: ' + relative)
print('PASS: all frozen REAP Python source hashes match')
