from typing import Callable, Tuple, Dict, List, Any
from glob import glob
import random
import os
import struct
import json

from torch import Tensor
from torchvision.datasets.vision import VisionDataset
import torch

from PIL import Image, ImageDraw, ImageOps
import numpy as np

from tqdm import tqdm


class OLHWDB2Line(VisionDataset):
    '''OLHWDB2 Dataset

    Args:
        root: Root directory of the dataset.
        characters_file: The path of character-set file.
        split: The dataset split, supports `"train"` (default) or `"test"`.
        align_size: Image alignment.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''

    CHARACTERS_FILE  = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'characters/ch_sym_tra.txt')

    def __init__(
            self,
            root:             str,
            characters_file:  str | None=None,
            split:            str='train',
            align_size:       int=32,
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        assert split in ('train', 'test')
        super().__init__(root, transforms, transform, target_transform)
        self.align_size = align_size

        characters_file = characters_file or self.CHARACTERS_FILE
        with open(characters_file, encoding='utf-8') as f:
            self.characters = ['', ' '] + sorted(
                set(f.read().replace(' ', '').replace('\n', '')))

        print('preload dataset...')
        lines = []
        self.paths = sorted(glob(os.path.join(root, split, '*.wptt')))
        for path in tqdm(self.paths, ncols=80):
            lines.extend(self._preload(path))
        self.lines = sorted(lines, key=lambda line: len(line['label']))
        self.min_label_size = len(self.lines[0]['label'])
        self.max_label_size = len(self.lines[-1]['label'])

    def _load_lines(self, filename:str) -> Tuple[Dict[str, Any], List[Dict]]:
        header = {}
        ZERO_TAG = b'\x00'.decode()
        with open(filename, 'rb') as f:
            bf_size_of_header = f.read(4)
            header['size_of_header'] = struct.unpack('<i', bf_size_of_header)[0]
            header['format_code'] = f.read(8).decode()
            illustration = b''
            while True:
                c = f.read(1)
                illustration += c
                if c == b'\0': break
            illustration = illustration.decode()
            header['code_type'] = f.read(20)
            code_length = struct.unpack('<H', f.read(2))[0]
            header['data_type'] = f.read(20).decode()
            header['sample_length'] = struct.unpack('<i', f.read(4))[0]
            header['page_index'] = struct.unpack('<i', f.read(4))[0]
            stroke_number = struct.unpack('<i', f.read(4))[0]
            strokes = []
            for i in range(stroke_number):
                point_number = struct.unpack('<H', f.read(2))[0]
                points = np.zeros((point_number, 2), dtype=np.int32)
                for j in range(point_number):
                    points[j, 0] = struct.unpack('<H', f.read(2))[0]
                    points[j, 1] = struct.unpack('<H', f.read(2))[0]
                strokes.append(points)
            line_number = struct.unpack('<H', f.read(2))[0]
            lines = []
            for i in range(line_number):
                line_stroke_number = struct.unpack('<H', f.read(2))[0]
                line_strokes = []
                for j in range(line_stroke_number):
                    line_stroke_index = struct.unpack('<H', f.read(2))[0]
                    line_strokes.append(strokes[line_stroke_index])
                line_char_number = struct.unpack('<H', f.read(2))[0]
                tag_code = f.read(code_length * line_char_number)
                tag_code = tag_code.replace(b'\xff\xff', b'#')
                tag_code_decoded = tag_code.decode('gb2312', errors='ignore')
                tag_code_decoded = tag_code_decoded.replace(ZERO_TAG, '')
                lines.append(dict(strokes=line_strokes, label=tag_code_decoded))
        return header, lines

    def _valid_label(self, label:str) -> bool:
        for c in label:
            if c not in self.characters:
                return False
        return True

    def _preload(self, path:str) -> List[Dict[str, Any]]:
        _, lines = self._load_lines(path)
        valided = []
        for line in lines:
            label = line['label']
            label = label.replace('―', '-')
            label = label.replace('×', 'x')
            label = label.replace('４', '4')
            # label = label.replace('‘', "'").replace('’', "'")
            if not self._valid_label(label): continue
            line['label'] = label
            valided.append(line)
        return valided

    def _draw_strokes(
            self,
            strokes: List,
            lw:      int=2,
            bg=0,
            fg=255,
            align:   int=96,
        ) -> Image.Image:

        min_x, min_y = strokes[0][0]
        max_x, max_y = min_x, min_y
        for stroke in strokes:
            for x, y in stroke:
                if x > max_x: max_x = x
                elif x < min_x: min_x = x
                if y > max_y: max_y = y
                elif y < min_y: min_y = y
        src_w, src_h = max_x - min_x, max_y - min_y
        dst_h, scale = align, align / src_h
        dst_w = round(scale * src_w)
        polygons = []
        for stroke in strokes:
            polygon = []
            for x, y in stroke:
                polygon.append((
                    round(scale * (x - min_x)),
                    round(scale * (y - min_y))))
            polygons.append(polygon)
        image = Image.new('L', (int(dst_w), int(dst_h)), color=bg)
        draw = ImageDraw.Draw(image)
        for polygon in polygons:
            draw.line(polygon, fill=fg, width=lw)
        return image

    def _load_image(self, strokes:List) -> Image.Image:
        lw = random.randint(2, 3)
        image = self._draw_strokes(strokes, lw=lw)
        src_w, src_h = image.size
        dst_h, scale = self.align_size, self.align_size / src_h
        dst_w = round(scale * src_w)
        image = image.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
        return image

    def _load_item(self, idx:int) -> Tuple[Image.Image, str]:
        line = self.lines[idx]
        strokes = line['strokes']
        image = self._load_image(strokes)
        return image, line['label']

    def __len__(self):
        return len(self.lines)

    def _item2label(self, idx:int):
        image, text = self._load_item(idx)
        image = ImageOps.invert(image)
        w, h = image.size
        points = [[0, 0], [w, 0], [w, h], [0, h]]
        label = {
            'transcription': text, 
            'points': points, 
            'language': '', 
            'illegibility': False,
        }
        return image, label

    def labels2lines(self, output:str):
        images_dir = os.path.join(output, 'images')
        os.makedirs(images_dir, exist_ok=True)
        gt_dict = dict()
        total = len(self)
        for idx in tqdm(range(total), ncols=80):
            image, label = self._item2label(idx)
            name = f'gt_{idx}'
            gt_dict[name] = [label]
            image_path = os.path.join(images_dir, name + '.jpg')
            image.save(image_path)
        gt_dict_path = os.path.join(output, 'labels.json')
        with open(gt_dict_path, 'w') as f:
            json.dump(gt_dict, f, ensure_ascii=False, indent=2)

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, Tensor]:
        image, text = self._load_item(idx)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.LongTensor(
            [self.characters.index(c) for c in text])
        return image, label
