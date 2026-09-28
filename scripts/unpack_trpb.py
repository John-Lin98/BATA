"""Extract only the frozen TrpB assay table from an independently obtained ALDE ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((Path(__file__).resolve().parents[1] / 'configs/TRPB_SOURCE.json').read_text())
    with zipfile.ZipFile(args.archive) as archive:
        info = archive.getinfo(source['archive_member'])
        if info.file_size != source['member_bytes']:
            raise ValueError('Unexpected member size')
        data = archive.read(info)
    if hashlib.sha256(data).hexdigest() != source['member_sha256']:
        raise ValueError('Frozen assay SHA mismatch')
    with args.output.open('xb') as stream:
        stream.write(data)
    print('PASS: extracted frozen TrpB CSV, no normalization or row filtering')


if __name__ == '__main__':
    main()
