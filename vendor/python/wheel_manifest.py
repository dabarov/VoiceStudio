"""Record the shipped wheel identities and hashes for independent auditing."""
import hashlib
import json
from pathlib import Path
import zipfile
from email.parser import BytesParser


def write(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    records = []
    for path in sorted((root / 'wheels').glob('*.whl')):
        with zipfile.ZipFile(path) as wheel:
            metadata = BytesParser().parsebytes(wheel.read(next(n for n in wheel.namelist() if n.endswith('.dist-info/METADATA'))))
        records.append({'path': 'wheels/' + path.name, 'name': metadata['Name'],
                        'version': metadata['Version'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (root / 'wheel-manifest.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    write()
