from typing import Callable, List, Any, Tuple
import random

from torch import Tensor
from torchvision.datasets.vision import VisionDataset
import torch

from PIL import Image

from .printing_character import PrintingCharacter
from .hwdb_gnt import HWDBGnt
from .hwdb_pot import HWDBPot


class TextlineGen(VisionDataset):
    '''Textline-Generator Dataset

    Args:
        root: Root directory of the dataset.
        fonts_dir: The directory of font files.
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

    def __init__(
            self,
            root:             str,
            fonts_dir:        str,
            characters_file:  str | None=None,
            split:            str='train',
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
            **kwargs,
        ):

        assert split in ('train', 'test')
        super().__init__(root, transforms, transform, target_transform)

        self._printing = PrintingCharacter(
            fonts_dir,
            characters_file,
            fontsize=kwargs.get('printing_fontsize', 48),
            limit=kwargs.get('printing_limit', 0))
        self._hwdb_rate = kwargs.get('hwdb_rate', 0.)
        self._hwdb_list = []
        if self._hwdb_rate > 0.:
            self._hwdb_list = [
                HWDBGnt(
                    kwargs.get('hwdb_gnt_dir'),
                    characters_file,
                    split=split,
                    limit=kwargs.get('hwdb_limit', 0)),
                HWDBPot(
                    kwargs.get('hwdb_pot_dir'),
                    characters_file,
                    split=split,
                    limit=kwargs.get('hwdb_limit', 0))]

        self.corpus = []
        text_length = kwargs.get('text_length', 10)
        print('loading corpus...')
        with open(root, encoding='utf-8') as f:
            while True:
                text = f.read(text_length)
                if len(text) < text_length: break
                text = text.strip().replace('\n', ' ')
                k = text_length - len(text)
                for _ in range(k):
                    text += random.choice(self._printing.characters)
                assert len(text) == text_length
                self.corpus.append(text)

        self.characters = ['', ' '] + self._printing.characters
        self.char2index = {c: i for i, c in enumerate(self.characters)}
        self._use_debug = kwargs.get('use_debug', False)
        self._reverse_rate = kwargs.get('reverse_rate', 0.)
        self._letter_spacing = kwargs.get('letter_spacing', 0.)
        self._layout_direction = kwargs.get('layout_direction', 'ltr')
        
    def __len__(self):
        return len(self.corpus)

    def _render_handwriting(
            self,
            text:      str,
            image:     Image.Image,
            xyxys:     List[Any],
            direction: str='ltb',
        ) -> Image.Image:

        output = Image.new('L', image.size)
        x_hwdb = random.choice(self._hwdb_list)
        for i, (x0, y0, x1, y1) in enumerate(xyxys):
            character = text[i]
            if character == ' ': continue
            index  = x_hwdb._char2index.get(character)
            anchor = [x0, y0]
            if index is not None:
                char_image, _ = x_hwdb[random.choice(index)]
                if direction == 'ttb':
                    char_image = char_image.transpose(Image.Transpose.ROTATE_90)
                img_w, img_h = char_image.size
                box_w, box_h = x1 - x0, y1 - y0
                if box_w < box_h:
                    dst_h = box_h
                    dst_w = max(1, min(box_w, round(dst_h / img_h * img_w)))
                    anchor[0] += (box_w - dst_w) // 2
                else:
                    dst_w = box_w
                    dst_h = max(1, min(box_h, round(dst_w / img_w * img_h)))
                    anchor[1] += (box_h - dst_h) // 2
                char_image = char_image.resize(
                    (dst_w, dst_h), resample=Image.Resampling.BILINEAR)
            else:
                char_image = image.crop((x0, y0, x1, y1))
            output.paste(char_image, anchor)
        return output

    def update_letter_spacing(
            self,
            image: Image.Image,
            xyxys: List[Any],
            size:  int,
        ) -> Image.Image:

        if size == 0: return image
        src_w, src_h = image.size
        exp_w = max(0, len(xyxys) - 1) * size + src_w
        expanded = Image.new('L', (exp_w, src_h), color=0)
        exp_size = 0
        for l, _, r, _ in xyxys:
            sub_image = image.crop((l, 0, r, src_h))
            expanded.paste(sub_image, (l + exp_size, 0, r + exp_size, src_h), mask=sub_image)
            exp_size += size
        return expanded

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, Tensor, int]:
        text = self.corpus[idx]

        font = self._printing.fonts[idx % len(self._printing.fonts)]
        text = font.random_lack(text, ['#'])

        printing, xyxys = font.text2image_with_xyxys(text, direction=self._layout_direction)

        applied_hwdb = self._hwdb_rate > max(1e-7, random.random())
        if applied_hwdb:
            image = self._render_handwriting(text, printing, xyxys, direction=self._layout_direction)
        else:
            image = printing

        letter_spacing = random.uniform(-self._letter_spacing, self._letter_spacing)
        if letter_spacing > -0.25:
            update_size = int(letter_spacing * image.size[1])
            image = self.update_letter_spacing(image, xyxys, update_size)

        reverse = int(self._reverse_rate > max(1e-7, random.random()))
        if reverse:
            image = image.transpose(Image.Transpose.ROTATE_180)
            text = text[::-1]

        if self.transform is not None:
            image = self.transform(image)

        target = torch.LongTensor([self.char2index[c] for c in text])

        if self._use_debug:
            return printing, image, target, reverse
        return image, target, reverse
