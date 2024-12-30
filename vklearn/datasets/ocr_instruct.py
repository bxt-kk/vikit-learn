from typing import List, Callable, Tuple, Sequence
from glob import glob
import os
import json
import random

from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
from torch import Tensor
import torch
from PIL import Image
import cv2 as cv
import numpy as np

from .ocr_synthesizer import OCRSynthesizer

from tqdm import tqdm


class OCRInstruct(Dataset):

    def __init__(
            self,
            subject_datas:    List[str | VisionDataset],
            synthesizer:      OCRSynthesizer,
            synthesis_rate:   float=1.,
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        self.subjects = []
        for data in subject_datas:
            if isinstance(data, str):
                self.subjects.append(InstructSubject(
                    data,
                    synthesizer.characters,
                    synthesizer._reverse_rate,
                    transforms=transforms,
                    transform=transform,
                    target_transform=target_transform,
                ))
            elif isinstance(data, Sequence) and isinstance(data[0], str):
                self.subjects.append(InstructSubject(
                    data[0],
                    synthesizer.characters,
                    synthesizer._reverse_rate,
                    transform=data[1],
                ))
            else:
                self.subjects.append(data)

        self.synthesizer   = synthesizer
        self.subject_total = sum([len(subject) for subject in self.subjects])

        self.synthesis_limit = int(len(synthesizer) * synthesis_rate)

    def __repr__(self):
        info = f'Dataset {self.__class__.__name__}\n'
        info += f'\tNumber of datapoints: {len(self)}\n'
        info += f'Synthesizer: {self.synthesizer}\n'
        info += 'Subjects:'
        for subject in self.subjects:
            info += f'\n* {str(subject)}'
        return info

    def __len__(self):
        return self.subject_total + self.synthesis_limit

    def __getitem__(self, idx:int):
        if idx >= self.subject_total:
            synthesis_size = len(self.synthesizer)
            if self.synthesis_limit == synthesis_size:
                return self.synthesizer[idx - self.subject_total]
            return self.synthesizer[random.randrange(synthesis_size)]
        begin = 0
        for subject in self.subjects:
            end = len(subject) + begin
            if idx < end: break
            begin = end
        return subject[idx - begin]


class InstructSubject(VisionDataset):

    def __init__(
            self,
            root:             str,
            characters:       List[str],
            reverse_rate:     float=0.,
            transforms:       Callable | None=None,
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
        ):

        super().__init__(root, transforms, transform, target_transform)

        self._reverse_rate = reverse_rate
        self.characters    = characters
        self.char2index    = {c: i for i, c in enumerate(self.characters)}

        print('preload subject dataset...')
        image_paths = sorted(glob(os.path.join(root, 'images/*.jpg')))
        with open(os.path.join(root, 'labels.json')) as f:
            labels = json.load(f)
        self.items = []
        for path in tqdm(image_paths, ncols=80):
            name = os.path.splitext(os.path.basename(path))[0]
            label = labels[name][0]
            if label['illegibility']: continue
            text = label['transcription'].strip()
            if not text: continue
            has_lack = False
            for c in text:
                if c in self.characters: continue
                has_lack = True
                break
            if has_lack: continue
            points = label['points']
            self.items.append((path, text, points))

    def __len__(self):
        return len(self.items)

    def _load_image(
            self,
            path:   str,
            points: List[List[int]],
        ) -> Image.Image:

        rect = cv.minAreaRect(np.intp(points))
        bbox = cv.boxPoints(rect)
        if bbox[1, 0] < bbox[3, 0]:
            width, height = rect[1]
        else:
            height, width = rect[1]
            bbox = bbox[[3, 0, 1, 2]]
        dst_pts = np.array([
            [0, height],
            [0, 0],
            [width, 0],
            [width, height]], dtype=np.float32)
        M = cv.getPerspectiveTransform(bbox, dst_pts)
        width = int(round(width))
        height = int(round(height))
        im_arr = cv.imread(path)
        im_arr = cv.warpPerspective(im_arr, M, (width, height), flags=cv.INTER_AREA)
        if len(im_arr.shape) == 2:
            im_arr = cv.cvtColor(im_arr, cv.COLOR_GRAY2RGB)
        elif len(im_arr.shape) == 3:
            im_arr = cv.cvtColor(im_arr, cv.COLOR_BGR2RGB)
        image = Image.fromarray(im_arr)

        # if image.size[0] < image.size[1]:
        #     image = image.transpose(Image.Transpose.ROTATE_90)
        return image

    def __getitem__(self, idx:int) -> Tuple[Image.Image | Tensor, Tensor, int]:
        path, text, points = self.items[idx]
        image = self._load_image(path, points)
        if (len(text) > 1) and (image.size[0] < image.size[1]):
            image = image.transpose(Image.Transpose.ROTATE_90)

        reverse = int(self._reverse_rate > max(1e-7, random.random()))
        if reverse:
            image = image.transpose(Image.Transpose.ROTATE_180)
            text = text[::-1]

        if self.transform is not None:
            image = self.transform(image)

        target = torch.LongTensor([self.char2index[c] for c in text])
        return image, target, reverse
