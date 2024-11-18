from typing import Callable, List, Tuple, Any
from glob import glob
import os
import math
import random

from torch import Tensor
from torchvision.datasets.vision import VisionDataset

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont
import numpy as np

from tqdm import tqdm


class Font:
    '''Printing Character Font

    Args:
        path: The font file path.
        size: Font size.
        chars: A list of characters.
    '''

    def __init__(
            self,
            path:  str,
            size:  int,
            chars: list,
        ):

        self._name    = os.path.basename(path)
        self._font    = ImageFont.truetype(path, size)
        self.valid    = self._font_checks(path, chars)
        self._lengths = dict()

    def _has_char(self, font:TTFont, char:str) -> bool:
        for table in font['cmap'].tables:
            if ord(char) in table.cmap.keys():
                return True
        return False

    def _font_checks(self, font_path:str, chars:List[str]) -> List[str]:
        font  = TTFont(font_path)
        valid = set(chars)
        for char in chars:
            if not self._has_char(font, char):
                valid.remove(char)
        return sorted(valid)

    def random_lack(
            self,
            text:  str,
            chars: List[str] | None=None,
        ) -> str:

        chars = chars or self.valid
        _text = ''
        for c in text:
            if c == ' ' or c in self.valid: _text += c
            else: _text += random.choice(chars)
        return _text

    def text2image_with_anchor(
            self,
            text:      str,
            direction: str='ltr',
        ) -> Tuple[Image.Image, Tuple[int, int]]:

        l, t, r, d = self._font.getbbox(text, direction=direction)
        anchor = (-l, -t)
        image  = Image.new('L', (r - l, d - t), color=0)
        draw   = ImageDraw.Draw(image)
        draw.text(anchor, text, font=self._font, fill=255, align='center', direction=direction)
        if direction == 'ttb':
            image = image.transpose(Image.Transpose.ROTATE_90)
            anchor = anchor[::-1]
        return image, anchor

    def get_length(
            self,
            text:      str,
            direction: str='ltr',
        ) -> int:

        length = self._lengths.get(text)
        if length is None:
            length = self._font.getlength(text, direction=direction)
            self._lengths[text] = length
        return length

    def text2image_with_xyxys(
            self,
            text:      str,
            direction: str='ltr',
        ) -> Tuple[Image.Image, List[Any]]:

        image, anchor = self.text2image_with_anchor(text, direction=direction)
        xyxys   = []
        lengths = [self.get_length(c, direction=direction) for c in text]
        bitmap  = np.asarray(image, dtype=np.uint8)

        left = 0
        for i, length in enumerate(lengths):
            right = left + length
            if i == 0: right += anchor[0]

            col_l = math.floor(left)
            col_r = math.ceil(right)
            if text[i] != ' ':
                submap = bitmap[:, col_l:col_r]
                shrink = max(1, submap.shape[1] // 9)
                for r0 in range(submap.shape[0]):
                    if submap[r0, shrink:-shrink].sum() != 0: break
                for r1 in range(submap.shape[0] - 1, 0, -1):
                    if submap[r1, shrink:-shrink].sum() != 0: break
                for c0 in range(submap.shape[1]):
                    if submap[:, c0].sum() != 0: break
                for c1 in range(submap.shape[1] - 1, 0, -1):
                    if submap[:, c1].sum() != 0: break
                x0 = col_l + c0
                x1 = col_l + c1
                y0 = r0
                y1 = r1
                if x1 - x0 < 2:
                    x0 -= 1
                    x1 = x0 + 2
                if y1 - y0 < 2:
                    y0 -= 1
                    y1 = y0 + 2
            else:
                x0 = col_l
                x1 = col_r
                y0 = 0
                y1 = bitmap.shape[0]
            xyxys.append([x0, y0, x1, y1])

            left = right
        return image, xyxys


class PrintingCharacter(VisionDataset):
    '''Printing Character Dataset

    Args:
        root: Root directory of the dataset.
        characters_file: The path of character-set file.
        fontsize: The size of font, default is 48.
        limit: Limit the number of font files to be loaded.
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
            fontsize:         int=48,
            limit:            int=0,
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        super().__init__(root, transforms, transform, target_transform)
        characters_file = characters_file or self.CHARACTERS_FILE
        with open(characters_file, encoding='utf-8') as f:
            self.characters = sorted(set(f.read().replace(' ', '').replace('\n', '')))

        font_paths = glob(os.path.join(root, '*/*'))
        if limit > 0: font_paths = font_paths[:limit]
        self.fonts = []
        print('loading fonts...')
        for font_path in tqdm(sorted(font_paths), ncols=80):
            self.fonts.append(Font(font_path, fontsize, self.characters))

    def __len__(self):
        return len(self.fonts) * len(self.characters)

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, int]:
        font_id      = idx % len(self.fonts)
        font         = self.fonts[font_id]
        character_id = idx // len(self.fonts) % len(font.valid)
        character    = font.valid[character_id]
        image        = font.text2image_with_anchor(character)[0]
        if self.transform is not None:
            image = self.transform(image)
        return image, character_id
