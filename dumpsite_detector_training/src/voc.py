"""VOC IO. Coordinates remain as stored; crop slices are [ymin:ymax,xmin:xmax]."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

CLASSES = ('domestic garbage', 'construction waste', 'agriculture forestry')
KEYS = ('xmin', 'ymin', 'xmax', 'ymax')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def objects(root):
    return [(o.findtext('name'), tuple(int(float(o.findtext('bndbox/' + k))) for k in KEYS))
            for o in root.findall('object')]

def parse(path):
    return ET.parse(path).getroot()

def set_box(root, index, box):
    targets = root.findall('object')
    if not 0 <= index < len(targets):
        raise ValueError(f'Object index {index} is out of range')
    for key, value in zip(KEYS, box):
        targets[index].find('bndbox/' + key).text = str(int(value))

def write(root, destination, filename=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for tag in ('path', 'folder'):
        for element in root.findall(tag):
            root.remove(element)
    if filename:
        element = root.find('filename')
        if element is None:
            element = ET.SubElement(root, 'filename')
        element.text = filename
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)

def validate(root):
    width = int(root.findtext('size/width'))
    height = int(root.findtext('size/height'))
    for name, (x1, y1, x2, y2) in objects(root):
        if name not in CLASSES:
            raise ValueError(f'Unknown class: {name}')
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Invalid bbox: {(x1, y1, x2, y2)}, size={width,height}')

def read_ids(path):
    result = Path(path).read_text(encoding='utf-8-sig').split()
    if any(Path(x).name != x or '/' in x or '\\' in x for x in result):
        raise ValueError('Image IDs must not contain paths')
    return result
