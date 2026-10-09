"""Predict paired optical/SAR tiles; save the same predictions on both modalities."""
import argparse
from pathlib import Path
from common import ROOT, SUFFIXES, sar_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True, help='Optical tile or directory inside images/.')
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', default=str(ROOT / 'runs' / 'predict'))
    args = parser.parse_args()
    import cv2
    import numpy as np
    from ultralytics import YOLO
    if not Path(args.weights).is_file():
        raise FileNotFoundError(args.weights)
    model = YOLO(args.weights, task='obb')
    source = Path(args.source).resolve()
    files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.suffix.lower() in SUFFIXES)
    if not files:
        raise FileNotFoundError(args.source)
    for optical_path in files:
        counterpart = sar_path(optical_path)
        optical, sar = cv2.imread(str(optical_path)), cv2.imread(str(counterpart))
        if optical is None or sar is None or optical.shape != sar.shape:
            raise ValueError(f'Invalid or mismatched pair: {optical_path}')
        result = model.predict(source=np.concatenate((optical, sar), axis=2), ch=6,
                               imgsz=512, device=args.device, conf=args.conf,
                               save=False, verbose=False)[0]
        relative = Path(*optical_path.parts[optical_path.parts.index('images') + 1:])
        for modality, image in (('optical', optical), ('sar', sar)):
            output = Path(args.output) / modality / relative.with_suffix('.jpg')
            output.parent.mkdir(parents=True, exist_ok=True)
            result.orig_img = image
            result.save(filename=str(output))
        label = (Path(args.output) / 'labels' / relative).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        result.save_txt(str(label), save_conf=True)
