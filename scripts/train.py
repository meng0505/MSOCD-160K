"""Train a main-component DASO-Det configuration on paired MSOCD-160K images."""
import argparse
from common import MODELS, ROOT, dataset_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--model', choices=MODELS, default='daso-det')
    parser.add_argument('--device', default='0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weights', help='Optional initialization checkpoint; omitted means training from scratch.')
    parser.add_argument('--name', help='Output run name.')
    parser.add_argument('--no-amp', action='store_true')
    args = parser.parse_args()
    from ultralytics import YOLO
    data = dataset_yaml(args.data_root)
    model = YOLO(str(MODELS[args.model]), task='obb')
    if args.weights:
        model.load(args.weights)
    model.train(data=data, ch=6, imgsz=512, epochs=args.epochs, batch=args.batch,
                device=args.device, workers=args.workers, seed=args.seed, optimizer='SGD',
                lr0=0.01, close_mosaic=10, patience=50, amp=not args.no_amp,
                cache=False, pretrained=False, project=str(ROOT / 'runs' / 'train'),
                name=args.name or args.model)


if __name__ == '__main__':
    main()
