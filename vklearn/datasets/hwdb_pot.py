from typing import Callable, Tuple, List
from glob import glob
from collections import defaultdict
import os
import struct
import random

from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from PIL import Image, ImageDraw
from tqdm import tqdm


class HWDBPot(VisionDataset):
    '''HWDB-Pot Dataset

    Args:
        root: Root directory of the dataset.
        characters_file: The path of character-set file.
        split: The dataset split, supports `"train"` (default) or `"test"`.
        limit: Limit the number of data files to be loaded.
        align_size: Image alignment.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''

    CHARACTERS_FILE  = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'characters_ec2.txt')

    def __init__(
            self,
            root:             str,
            characters_file:  str | None=None,
            split:            str='train',
            limit:            int=0,
            align_size:       int=96,
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        assert split in ('train', 'test')
        super().__init__(root, transforms, transform, target_transform)
        pot_files = glob(os.path.join(root, f'*{split}', '*.pot'))
        self._build_index(pot_files, characters_file, limit)
        self._align_size = align_size

    def _load_characters(self, characters_file:str) -> List[str]:
        with open(characters_file, encoding='utf-8') as f:
            return sorted(set(f.read().replace(' ', '').replace('\n', '')))

    def _build_index(
            self,
            pot_files:       list,
            characters_file: str | None,
            limit:           int,
        ):

        char2index       = defaultdict(list)
        index            = []
        valid_characters = set()
        characters = self._load_characters(characters_file or self.CHARACTERS_FILE)
        pot_files_sorted = sorted(pot_files)
        if limit > 0:
            pot_files_sorted = pot_files_sorted[:limit]
        print('build index for pot files...')
        for pot_i, pot_file in enumerate(tqdm(pot_files_sorted, ncols=80)):
            with open(pot_file, 'rb') as f:
                while True:
                    pointer = f.tell()
                    sample_size_bytes = f.read(2)
                    if sample_size_bytes == b'': break
                    sample_size = struct.unpack('<H', sample_size_bytes)[0]
                    target_code = f.read(4)[::-1].decode('gbk', 'ignore').strip('\x00')
                    if target_code in characters:
                        char2index[target_code].append(len(index))
                        index.append([pot_i, pointer, target_code])
                        valid_characters.add(target_code)
                    f.seek(sample_size - 2 - 4, 1)
        self._char2index = char2index
        self._data_index = index
        self._pot_files  = pot_files_sorted
        self.characters  = sorted(valid_characters)

    def __len__(self):
        return len(self._data_index)

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, int]:
        pot_i, pointer, character = self._data_index[idx]
        with open(self._pot_files[pot_i], 'rb') as f:
            f.seek(pointer + 2, 0)
            target_code = f.read(4)[::-1].decode('gbk', 'ignore').strip('\x00')
            assert target_code == character
            stroke_number = struct.unpack('<H', f.read(2))[0]
            strokes = []
            coord_l = 100000
            coord_t = 100000
            coord_r = 0
            coord_b = 0
            for stroke_i in range(stroke_number):
                stroke = []
                for coordinate_i in range(10000):
                    coordinate = struct.unpack('<2h', f.read(4))
                    if coordinate[0] == -1 and coordinate[1] == 0: break
                    coord_l = min(coordinate[0], coord_l)
                    coord_t = min(coordinate[1], coord_t)
                    coord_r = max(coordinate[0], coord_r)
                    coord_b = max(coordinate[1], coord_b)
                    stroke.append(coordinate)
                strokes.append(stroke)
        src_w, src_h = coord_r - coord_l, coord_b - coord_t
        scale = self._align_size / max(src_w, src_h)
        dst_w, dst_h = round(scale * src_w), round(scale * src_h)
        image = Image.new('L', (dst_w + 4, dst_h + 4), color=0)
        draw = ImageDraw.Draw(image)
        for stroke in strokes:
            polygon = []
            for x, y in stroke:
                polygon.append((
                    round(scale * (x - coord_l + 2)),
                    round(scale * (y - coord_t + 2))))
            stroke_width = random.randint(2, 3)
            draw.line(polygon, fill=255, width=stroke_width)
        if self.transform is not None:
            image = self.transform(image)
        character_id = self.characters.index(character)
        return image, character_id
