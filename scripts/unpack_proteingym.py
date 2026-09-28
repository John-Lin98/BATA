"""Extract only paper-used ProteinGym CSVs/reference FASTAs from user-obtained files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bata.data import sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--archive',type=Path,required=True)
parser.add_argument('--metadata',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
manifest=json.loads((Path(__file__).resolve().parents[1]/'configs/PROTEINGYM_SOURCE.json').read_text())
if args.output.exists():parser.error('Use a new output directory')
if sha256(args.archive)!=manifest['archive_sha256'] or sha256(args.metadata)!=manifest['metadata_sha256']:
    parser.error('Input files do not match the frozen ProteinGym v1.3 archive/metadata')
with args.metadata.open(newline='') as f:metadata={row['DMS_id']:row for row in csv.DictReader(f)}
with zipfile.ZipFile(args.archive) as archive:
    for benchmark,item in manifest['datasets'].items():
        seq=metadata[item['dataset_id']]['target_seq']
        if hashlib.sha256(seq.encode()).hexdigest()!=item['reference_sequence_sha256']:
            parser.error('Reference identity mismatch: '+benchmark)
        if item['archive_member'] not in archive.namelist():parser.error('Missing archive member')
    args.output.mkdir(parents=True,exist_ok=False)
    for benchmark,item in manifest['datasets'].items():
        output=args.output/(benchmark+'.csv')
        # Never extract arbitrary archive paths; output names come from fixed manifest keys.
        with archive.open(item['archive_member']) as source,output.open('xb') as target:
            shutil.copyfileobj(source,target)
        if sha256(output)!=item['sha256']:raise RuntimeError('Extracted source mismatch: '+benchmark)
        with (args.output/(benchmark+'.fasta')).open('x') as f:
            f.write('>'+item['dataset_id']+'\n'+metadata[item['dataset_id']]['target_seq']+'\n')
print('PASS: three frozen assay files and reference FASTAs extracted locally; no labels analyzed')
