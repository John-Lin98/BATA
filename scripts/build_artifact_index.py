"""Index lightweight publication assets; refuses unclassified files and raw artifacts."""
import csv
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    'bata': ('method implementation', 'Main BATA framework / necessary baseline and mechanism adapters'),
    'scripts': ('reproduction tooling', 'Data preparation / experiment entry points / evaluation / aggregation'),
    'tests': ('engineering validation', 'Protocol and extraction correctness; not scientific performance evidence'),
    'configs': ('external asset identity', 'Frozen representation and upstream data acquisition'),
    'manifests': ('frozen reproduction configuration', 'Paper cohorts / external dependency identity / scoped parity evidence'),
    'results': ('paper result summary', 'Main tables / sensitivity / mechanisms / appendix including negative results'),
    'figures': ('figure source', 'Paper figures and their source-data mappings'),
    'paper_figures': ('final figure reproduction', 'Frozen plotting inputs and scripts for all final paper figures'),
    'docs': ('reader documentation', 'Method execution, results, limitations and source provenance'),
    'licenses': ('third-party attribution', 'Required notices for included attributed code'),
}
ROOT_FILES = {'.gitignore', '.gitattributes', 'README.md', 'EXCLUDE_INDEX.csv', 'requirements.txt',
              'requirements-baselines.txt', 'requirements-features.txt', 'LICENSE',
              'CITATION.cff'}


def main():
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    rows = []
    for name in sorted(set(names.decode().split('\0')) - {'', 'PAPER_ARTIFACT_INDEX.csv'}):
        relative = Path(name)
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Expected regular publication file: ' + name)
        if relative.parts[0] in CATEGORIES:
            kind, relation = CATEGORIES[relative.parts[0]]
        elif name in ROOT_FILES:
            kind, relation = 'release metadata', 'Repository use / environment / release boundary'
        else:
            raise ValueError('Unclassified asset: ' + name)
        if path.suffix in {'.pt', '.pkl', '.joblib', '.npy', '.npz', '.safetensors', '.zip'} or path.stat().st_size > 20*1024*1024:
            raise ValueError('Unexpected raw/binary/large asset: ' + name)
        content = path.read_bytes()
        rows.append(dict(path=name, asset_class=kind, paper_relation=relation,
                         bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                         disposition='include', note='File inventory only; not a claim of complete validation'))
    with (ROOT / 'PAPER_ARTIFACT_INDEX.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(f'Indexed {len(rows)} files / {sum(r["bytes"] for r in rows)} bytes; index excludes itself')


if __name__ == '__main__':
    main()
