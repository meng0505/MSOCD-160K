"""Measure unfused parameters and THOP FLOPs; optionally time forward-only GPU inference."""
import argparse
import copy
import json
from common import MODELS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all'] + list(MODELS), default='all')
    parser.add_argument('--device', default='cpu', help='cpu or CUDA device index, e.g. 0.')
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--timing', action='store_true', help='Forward only, batch 1; excludes preprocessing and NMS.')
    parser.add_argument('--warmup', type=int, default=30)
    parser.add_argument('--iterations', type=int, default=200)
    args = parser.parse_args()
    import torch
    import thop
    from ultralytics.nn.tasks import OBBModel
    torch.set_num_threads(2)
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    names = list(MODELS) if args.model == 'all' else [args.model]
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError('iterations must be positive and warmup nonnegative.')
    for name in names:
        model = OBBModel(str(MODELS[name]), ch=6, nc=6, verbose=False).eval().to(device)
        x = torch.zeros(1, 6, args.imgsz, args.imgsz, device=device)
        params = sum(p.numel() for p in model.parameters())
        with torch.inference_mode():
            # THOP attaches counters; profile a copy so timing sees an untouched model.
            macs, _ = thop.profile(copy.deepcopy(model), inputs=(x,), verbose=False)
        row = dict(model=name, params=params, params_M=params / 1e6, FLOPs_G=2*macs/1e9,
                   input=list(x.shape), torch_version=torch.__version__, device=str(device))
        if args.timing:
            if device.type != 'cuda':
                raise ValueError('Timing is GPU-only. Use --device 0.')
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(x)
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    model(x)
                end.record()
                torch.cuda.synchronize()
            row['forward_ms'] = start.elapsed_time(end) / args.iterations
            row['forward_FPS'] = 1000 / row['forward_ms']
            row['GPU'] = torch.cuda.get_device_name(device)
            row['precision'] = 'FP32'
        print(json.dumps(row))
    print('FLOPs = 2 x THOP MACs; functional FFT, masks and elementwise operations are not fully counted.')
