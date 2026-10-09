"""Portable dataset/config paths shared by the release command-line tools."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = {name: ROOT / 'ultralytics' / 'cfg' / 'models' / (name + '.yaml')
          for name in ('baseline', 'smdm', 'scim', 'daso-det')}
NAMES = ['plane_SO-A', 'plane_S-only', 'plane_O-only',
         'ship_SO-A', 'ship_S-only', 'ship_O-only']
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def dataset_root(value):
    root = Path(value).expanduser().resolve()
    for folder in ('images', 'imagesSAR', 'labels'):
        for split in ('train', 'test'):
            if not (root / folder / split).is_dir():
                raise FileNotFoundError(str(root / folder / split))
    return root


def dataset_yaml(root):
    import yaml
    root = dataset_root(root)
    output = ROOT / 'runs' / 'data' / 'msocd.yaml'
    output.parent.mkdir(parents=True, exist_ok=True)
    config = dict(path=root.as_posix(), train='images/train', val='images/test',
                  test='images/test', names=dict(enumerate(NAMES)), modality_dir='imagesSAR')
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    return str(output)


def sar_path(optical):
    parts = list(Path(optical).resolve().parts)
    if 'images' not in parts:
        raise ValueError('Optical source must be inside a directory named images.')
    parts[parts.index('images')] = 'imagesSAR'
    counterpart = Path(*parts)
    if not counterpart.is_file():
        raise FileNotFoundError(str(counterpart))
    return counterpart
