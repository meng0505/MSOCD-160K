"""Evaluate a trained six-class paired OBB detector."""
import argparse
from pathlib import Path
from common import ROOT, dataset_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    args = parser.parse_args()
    if not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)
    from ultralytics import YOLO
    model = YOLO(args.weights, task='obb')
    model.val(data=dataset_yaml(args.data_root), ch=6, imgsz=512, split=args.split,
              batch=args.batch, workers=args.workers, device=args.device,
              project=str(ROOT / 'runs' / 'val'), name=Path(args.weights).stem)
