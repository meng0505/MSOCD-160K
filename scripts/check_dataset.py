"""Check all pairs and joint OBB labels before training."""
import argparse
import json
import math
from common import NAMES, SUFFIXES, dataset_root


def check(root, image_sizes=False):
    result = {}
    for split in ('train', 'test'):
        optical = root / 'images' / split
        sar = root / 'imagesSAR' / split
        labels = root / 'labels' / split
        files = sorted(p for p in optical.rglob('*') if p.suffix.lower() in SUFFIXES)
        optical_names = {p.relative_to(optical) for p in files}
        sar_names = {p.relative_to(sar) for p in sar.rglob('*') if p.suffix.lower() in SUFFIXES}
        if optical_names != sar_names:
            raise ValueError(f'{split}: unequal optical/SAR file sets ({len(optical_names ^ sar_names)} mismatches).')
        counts = [0] * len(NAMES)
        for image in files:
            rel = image.relative_to(optical)
            label = (labels / rel).with_suffix('.txt')
            if not label.is_file():
                raise FileNotFoundError(str(label))
            for number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 9:
                    raise ValueError(f'{label}:{number}: expected class ID and 8 normalized OBB coordinates.')
                cls = int(fields[0])
                values = [float(v) for v in fields[1:]]
                if not (0 <= cls < 6) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise ValueError(f'{label}:{number}: invalid class or coordinates.')
                counts[cls] += 1
            if image_sizes:
                from PIL import Image
                with Image.open(image) as opt, Image.open(sar / rel) as other:
                    if opt.size != other.size or opt.size != (512, 512):
                        raise ValueError(f'{image}: expected a paired 512 x 512 tile.')
        result[split] = dict(paired_tiles=len(files), object_state_instances=sum(counts),
                             per_class=dict(zip(NAMES, counts)))
    total = [sum(result[s]['per_class'][n] for s in result) for n in NAMES]
    result['total'] = dict(object_state_instances=sum(total), associated_pairs=total[0]+total[3])
    # Count actual per-modality annotation files, rather than assuming every clipped
    # associated target still has two boxes after patch generation.
    per_modality = {}
    for folder in ('labelsOpt', 'labelsSAR'):
        if (root / folder).is_dir():
            per_modality[folder] = sum(sum(bool(line.strip()) for line in p.read_text(encoding='utf-8-sig').splitlines())
                                      for p in (root / folder).rglob('*.txt'))
    if len(per_modality) == 2:
        result['total']['modality_specific_obbs'] = sum(per_modality.values())
        result['total']['per_modality_obbs'] = per_modality
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--check-image-sizes', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(dataset_root(args.data_root), args.check_image_sizes), indent=2))
