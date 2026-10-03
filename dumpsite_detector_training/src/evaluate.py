"""Score predictions: confidence >=0.1, same-class IoU >=0.5, greedy one-to-one matching.

Input CSV columns: image_id,class,score,xmin,ymin,xmax,ymax.
All prediction coordinates must be in ORIGINAL image coordinates and the same
continuous xyxy convention as the input annotations. NMS belongs to the detector.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from .voc import CLASSES, objects, parse, read_ids

def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def score(voc, split, predictions, threshold=0.1, iou_threshold=0.5):
    ids = read_ids(split); allowed = set(ids)
    pred = defaultdict(list)
    for row in predictions:
        if row['image_id'] not in allowed or row['class'] not in CLASSES:
            raise ValueError('Unknown image or class in predictions')
        confidence = float(row['score'])
        box = tuple(float(row[k]) for k in ('xmin','ymin','xmax','ymax'))
        import math
        if not all(math.isfinite(x) for x in (*box, confidence)) or not 0 <= confidence <= 1 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid predicted score or bbox')
        if confidence >= threshold:
            pred[row['image_id']].append((row['class'], confidence, box))
    totals = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in CLASSES}
    for image_id in ids:
        gt = objects(parse(Path(voc)/'Annotations'/(image_id+'.xml')))
        matched = set()
        for c, _, box in sorted(pred[image_id], key=lambda x: -x[1]):
            choices = [(iou(box, b), j) for j, (name, b) in enumerate(gt) if name == c and j not in matched]
            overlap, j = max(choices, default=(0.0, -1))
            if j >= 0 and overlap >= iou_threshold:
                matched.add(j); totals[c]['TP'] += 1
            else: totals[c]['FP'] += 1
        for j, (c, _) in enumerate(gt):
            if j not in matched: totals[c]['FN'] += 1
    for values in totals.values():
        t, f, n = values['TP'], values['FP'], values['FN']
        values['precision'] = t/(t+f) if t+f else 0.0
        values['recall'] = t/(t+n) if t+n else 0.0
    t = sum(x['TP'] for x in totals.values()); f = sum(x['FP'] for x in totals.values()); n = sum(x['FN'] for x in totals.values())
    return {'confidence_threshold': threshold, 'iou_threshold': iou_threshold,
            'coordinate_convention': 'continuous xyxy; no inclusive +1',
            'classes': totals, 'macro_precision': sum(x['precision'] for x in totals.values())/len(CLASSES),
            'macro_recall': sum(x['recall'] for x in totals.values())/len(CLASSES),
            'micro_precision': t/(t+f) if t+f else 0.0, 'micro_recall': t/(t+n) if t+n else 0.0}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voc',required=True);p.add_argument('--split',required=True)
    p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    with open(a.predictions,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    report=score(a.voc,a.split,rows)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
