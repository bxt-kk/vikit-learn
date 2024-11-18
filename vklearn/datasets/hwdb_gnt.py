from typing import Callable, List, Tuple
from glob import glob
from collections import defaultdict
import os
import struct

from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from PIL import Image
from tqdm import tqdm
import numpy as np


class HWDBGnt(VisionDataset):
    '''HWDB-Gnt Dataset

    Args:
        root: Root directory of the dataset.
        split: The dataset split, supports `"train"` (default) or `"test"`.
        characters_file: The path of character-set file.
        limit: Limit the number of data files to be loaded.
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
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        assert split in ('train', 'test')
        super().__init__(root, transforms, transform, target_transform)
        gnt_files = glob(os.path.join(root, f'*{split}', '*.gnt'))
        self._build_index(gnt_files, characters_file, limit)

    def _load_characters(self, characters_file:str) -> List[str]:
        with open(characters_file, encoding='utf-8') as f:
            return sorted(set(f.read().replace(' ', '').replace('\n', '')))

    def _build_index(
            self,
            gnt_files:       list,
            characters_file: str,
            limit:           int,
        ):

        char2index       = defaultdict(list)
        index            = []
        valid_characters = set()
        characters = self._load_characters(characters_file or self.CHARACTERS_FILE)
        gnt_files_sorted = sorted(gnt_files)
        if limit > 0:
            gnt_files_sorted = gnt_files_sorted[:limit]
        print('build index for gnt files...')
        for gnt_i, gnt_file in enumerate(tqdm(gnt_files_sorted, ncols=80)):
            with open(gnt_file, 'rb') as f:
                while True:
                    pointer = f.tell()
                    sample_size_bytes = f.read(4)
                    if sample_size_bytes == b'': break
                    sample_size = struct.unpack('<I', sample_size_bytes)[0]
                    target_code = f.read(2).decode('gbk', 'ignore').strip('\x00')
                    if target_code in characters:
                        char2index[target_code].append(len(index))
                        index.append([gnt_i, pointer, target_code])
                        valid_characters.add(target_code)
                    f.seek(sample_size - 4 - 2, 1)
        self._char2index = char2index
        self._data_index = index
        self._gnt_files  = gnt_files_sorted
        self.characters  = sorted(valid_characters)

    def __len__(self):
        return len(self._data_index)

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, int]:
        gnt_i, pointer, character = self._data_index[idx]
        with open(self._gnt_files[gnt_i], 'rb') as f:
            f.seek(pointer + 4, 0)
            target_code = f.read(2).decode('gbk', 'ignore').strip('\x00')
            assert target_code == character
            width  = struct.unpack('<H', f.read(2))[0]
            height = struct.unpack('<H', f.read(2))[0]
            buffer = f.read(width * height)
            bitmap = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
        val_min = bitmap.min()
        val_max = bitmap.max()
        bitmap  = (255 * (1 - (bitmap - val_min) / max(1, val_max - val_min))).astype(np.uint8)
        image = Image.fromarray(bitmap)
        if self.transform is not None:
            image = self.transform(image)
        character_id = self.characters.index(character)
        return image, character_id
