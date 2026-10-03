"""Training-only crop preparation, sampling and four-group dataset construction."""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from .voc import CLASSES, objects, parse, read_ids, set_box, validate, write, sha256

ANCHORS = [('af', 'agriculture forestry'), ('cw', 'construction waste'), ('do', 'domestic garbage')]
COUNTS = {'af': 1634, 'cw': 1447, 'do': 460}

def load_manifest(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def prepare(voc, split, output):
    voc, output = Path(voc), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with (output / 'metadata.jsonl').open('w', encoding='utf-8') as f:
        for image_id in read_ids(split):
            root = parse(voc / 'Annotations' / (image_id + '.xml'))
            validate(root)
            image = cv2.imread(str(voc / 'JPEGImages' / (image_id + '.jpg')))
            if image is None:
                raise FileNotFoundError(image_id)
            for index, (name, (x1, y1, x2, y2)) in enumerate(objects(root)):
                filename = f'{image_id}_{index}.jpg'
                # Use the declared xyxy slicing convention consistently.
                patch = image[y1:y2, x1:x2]
                if not cv2.imwrite(str(output / filename), patch):
                    raise IOError(filename)
                f.write(json.dumps({'file_name': filename, 'text': name}) + '\n')
                records.append(dict(image=image_id, idx=index, **{'class': name},
                                    xmin=x1, xmax=x2, ymin=y1, ymax=y2))
    with (output / 'record.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'class', 'xmin', 'xmax', 'ymin', 'ymax', 'idx'])
        w.writeheader(); w.writerows(records)

def make_plan(records_path, split, output, counts=None, seed=42):
    records = load_manifest(records_path)
    allowed = set(read_ids(split))
    if any(r['image'] not in allowed for r in records):
        raise ValueError('record.csv includes images outside the declared training split')
    rng = np.random.RandomState(seed)  # deterministic sampling sequence
    rows = []
    for code, name in ANCHORS:
        pool = [i for i, r in enumerate(records) if r['class'] == name]
        count = (counts or COUNTS)[code]
        if not pool and count:
            raise ValueError(f'No training objects for {name}')
        for i, choice in enumerate(rng.choice(pool, count, replace=True)):
            record = records[int(choice)]
            rows.append({'image_id': f'aug_{len(rows):04d}', 'source_id': record['image'],
                         'object_index': record['idx'], 'class': name,
                         'patch_file': f'{code.upper()}{i:03d}RD.jpg',
                         'generation_seed': i, 'transform_seed': seed + len(rows)})
    with open(output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def color_jitter(image, seed):
    """Two-of-three OpenCV color transforms with a local RNG."""
    rng = np.random.RandomState(seed)
    out = image.copy()
    for kind in rng.choice(['b', 's', 'c'], 2, replace=False):
        if kind in ('b', 's'):
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
            channel = 2 if kind == 'b' else 1
            delta = rng.randint(-50, 50)
            hsv[:, :, channel] = np.clip(hsv[:, :, channel].astype(np.int16) + delta, 0, 255).astype(np.uint8)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            contrast = rng.randint(40, 90)
            out = np.clip(out.astype(np.int16) * (contrast / 127 + 1) - contrast + 10, 0, 255).astype(np.uint8)
    return out

def compose(image, root, index, patch):
    """Replace one patch; preserve every other object's label and geometry."""
    validate(root)
    before = objects(root)
    if not 0 <= index < len(before):
        raise IndexError(index)
    x1, y1, x2, y2 = before[index][1]
    height, width = image.shape[:2]
    if (width, height) != (int(root.findtext('size/width')), int(root.findtext('size/height'))):
        raise ValueError('Image dimensions differ from XML')
    result = image.copy()
    result[y1:y2, x1:x2] = cv2.resize(patch, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
    updated = copy.deepcopy(root)
    set_box(updated, index, (x1, y1, x2, y2))
    if objects(updated) != before:
        raise AssertionError('A same-position replacement must preserve all object annotations')
    return result, updated

def build(voc, split, manifest, patches, output, group, eval_splits=()):
    voc, output = Path(voc), Path(output)
    ids = read_ids(split)
    if len(ids) != len(set(ids)):
        raise ValueError('Original training split contains duplicates')
    rows = load_manifest(manifest)
    allowed = set(ids)
    if any(r['source_id'] not in allowed for r in rows):
        raise ValueError('Augmentation source outside training split')
    new_ids = [r['image_id'] for r in rows]
    if len(new_ids) != len(set(new_ids)) or allowed.intersection(new_ids):
        raise ValueError('Augmentation IDs must be unique and distinct from original IDs')
    eval_ids = {}
    seen = set(ids)
    for name, path in eval_splits:
        values = read_ids(path)
        if len(values) != len(set(values)) or seen.intersection(values) or set(new_ids).intersection(values):
            raise ValueError('Training/evaluation splits overlap')
        eval_ids[name] = values; seen.update(values)
    output.mkdir(parents=True, exist_ok=False)
    for d in ('JPEGImages', 'Annotations', 'ImageSets/Main'):
        (output / d).mkdir(parents=True)
    for image_id in ids + [x for values in eval_ids.values() for x in values]:
        shutil.copy2(voc / 'JPEGImages' / (image_id + '.jpg'), output / 'JPEGImages' / (image_id + '.jpg'))
        root = parse(voc / 'Annotations' / (image_id + '.xml')); validate(root)
        write(root, output / 'Annotations' / (image_id + '.xml'), image_id + '.jpg')
    generated, trace = [], []
    for row in ([] if group == 'baseline' else rows):
        source_id, image_id = row['source_id'], row['image_id']
        if Path(image_id).name != image_id or any(c in image_id for c in '/\\'):
            raise ValueError('Invalid image ID')
        root = parse(voc / 'Annotations' / (source_id + '.xml'))
        index = int(row['object_index'])
        if objects(root)[index][0] != row['class']:
            raise ValueError('Manifest object index/class mismatch')
        source = voc / 'JPEGImages' / (source_id + '.jpg')
        image = cv2.imread(str(source))
        if image is None:
            raise FileNotFoundError(source)
        patch_hash = None
        if group == 'balanced':
            # Exact image bytes, with a new identity and matching XML; no pixel transformation.
            shutil.copy2(source, output / 'JPEGImages' / (image_id + '.jpg'))
        else:
            if group == 'sd_lora':
                patch_path = Path(patches) / row['patch_file']
                patch = cv2.imread(str(patch_path))
                if patch is None:
                    raise FileNotFoundError(f'Missing generated patch: {patch_path}')
                patch_hash = sha256(patch_path)
            else:
                x1, y1, x2, y2 = objects(root)[index][1]
                patch = color_jitter(image[y1:y2, x1:x2], int(row['transform_seed']))
            result, root = compose(image, root, index, patch)
            if not cv2.imwrite(str(output / 'JPEGImages' / (image_id + '.jpg')), result):
                raise IOError(image_id)
        write(root, output / 'Annotations' / (image_id + '.xml'), image_id + '.jpg')
        generated.append(image_id)
        trace.append(dict(row, group=group, patch_sha256=patch_hash, source_sha256=sha256(source)))
    (output / 'ImageSets/Main/train.txt').write_text('\n'.join(ids + generated) + '\n')
    for name, values in eval_ids.items():
        (output / 'ImageSets/Main' / (name + '.txt')).write_text('\n'.join(values) + '\n')
    (output / 'provenance.json').write_text(json.dumps(trace, indent=2), encoding='utf-8')
    print(json.dumps({'group': group, 'original_images': len(ids), 'added_images': len(generated)}))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='command', required=True)
    a = s.add_parser('prepare'); a.add_argument('--voc', required=True); a.add_argument('--split', required=True); a.add_argument('--output', required=True)
    a = s.add_parser('plan'); a.add_argument('--records', required=True); a.add_argument('--split', required=True); a.add_argument('--output', required=True); a.add_argument('--per-class', type=int)
    a = s.add_parser('build'); a.add_argument('--voc', required=True); a.add_argument('--split', required=True); a.add_argument('--manifest', required=True); a.add_argument('--patches', default='examples/patches'); a.add_argument('--output', required=True); a.add_argument('--group', choices=['baseline', 'balanced', 'color_jitter', 'sd_lora'], required=True); a.add_argument('--val-split'); a.add_argument('--test-split')
    a = p.parse_args()
    if a.command == 'prepare': prepare(a.voc, a.split, a.output)
    elif a.command == 'plan': make_plan(a.records, a.split, a.output, {k: a.per_class for k, _ in ANCHORS} if a.per_class is not None else None)
    else: build(a.voc, a.split, a.manifest, a.patches, a.output, a.group, [(k, getattr(a, k+'_split')) for k in ('val', 'test') if getattr(a, k+'_split')])

if __name__ == '__main__':
    main()
