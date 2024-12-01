from typing import List, Any, Dict, Iterable, Callable
import random
import math

from torch import Tensor
import torch
import torch.nn.functional as F

from torchvision import transforms
from torchmetrics.text import CharErrorRate #, WordErrorRate

from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
import cv2 as cv

from .basic import Basic


class OCR(Basic):

    def __init__(self, categories:List[str]):
        super().__init__()

        self.categories  = list(categories)
        self.num_classes = len(categories)

        self.cer_metric = CharErrorRate()
        # self.wer_metric = WordErrorRate()

        self._categorie_arr = np.array(self.categories)

    def preprocess(
            self,
            image:      Image.Image,
            align_size: int,
        ) -> Tensor:

        src_w, src_h = image.size
        scale = align_size / src_h
        dst_w, dst_h = round(src_w * scale), round(src_h * scale)
        resized = image.resize((dst_w, dst_h), resample=Image.Resampling.BILINEAR)
        if dst_w % dst_h == 0:
            return self._image2tensor(resized).unsqueeze(dim=0)
        dst_w = math.ceil(dst_w / align_size) * align_size
        aligned = Image.new(image.mode, (dst_w, dst_h))
        aligned.paste(resized, (0, 0))
        return self._image2tensor(aligned).unsqueeze(dim=0)

    def recognize(
            self,
            image:      Image.Image,
            top_k:      int=10,
            align_size: int=32,
        ) -> List[Dict[str, Any]]:
        assert not 'this is an empty func'

    def calc_loss(
            self,
            inputs:         Tensor,
            targets:        Tensor,
            input_lengths:  Tensor,
            target_lengths: Tensor,
            zero_infinity:  bool=False,
        ) -> Dict[str, Any]:

        log_probs = inputs.transpose(0, 1).log_softmax(dim=2) # T, N, C
        # batch, seq_len = log_probs.shape[1], log_probs.shape[0]
        # input_lengths = torch.full((batch, ), seq_len, dtype=torch.long).to(inputs.device)
        return dict(
            loss=F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                zero_infinity=zero_infinity,
                reduction='mean'),
        )

    def calc_score(
            self,
            inputs:         Tensor,
            targets:        Tensor,
            input_lengths:  Tensor,
            target_lengths: Tensor,
        ) -> Dict[str, Any]:

        preds = inputs.argmax(dim=2) # .cpu() # n, T

        # kernel = torch.tensor([[[-1, 1]]]).type_as(preds)
        # mask = torch.conv1d(
        #     F.pad(preds.unsqueeze(1), [0, 1], value=0), kernel).squeeze(1) != 0
        mask = (F.pad(preds, [0, 1], value=0)[:, 1:] - preds) != 0
        preds = preds * mask
        # preds, mask: n, T

        nonzero_mask = preds > 0
        sort_weights = torch.arange(nonzero_mask.shape[1], 0, step=-1, device=preds.device).unsqueeze(0)
        indices = torch.argsort(nonzero_mask * sort_weights, dim=1, descending=True)
        preds[~nonzero_mask] = self.num_classes
        preds = torch.gather(preds, dim=1, index=indices)
        common_length = min(preds.shape[1], targets.shape[1])
        compared = preds[:, :common_length] == targets[:, :common_length] # .cpu()
        max_lengths = torch.maximum(nonzero_mask.sum(dim=1), target_lengths) # .cpu())
        score = (compared.sum(dim=1) / torch.clamp_min(max_lengths, 1)).mean()
        return dict(
            rough_accuracy=score,
        )

    def update_metric(
            self,
            inputs:         Tensor,
            targets:        Tensor,
            input_lengths:  Tensor,
            target_lengths: Tensor,
        ):

        preds_tensor = inputs.argmax(dim=2) # .cpu() # n, T
        # kernel = torch.tensor([[[-1, 1]]]).type_as(preds_tensor)
        # mask = torch.conv1d(
        #     F.pad(preds_tensor.unsqueeze(1), [0, 1], value=0), kernel).squeeze(1) != 0
        mask = (F.pad(preds_tensor, [0, 1], value=0)[:, 1:] - preds_tensor) != 0
        preds_tensor = preds_tensor * mask

        preds = [''.join(items) for items in
            self._categorie_arr[preds_tensor.cpu().numpy()]]
        trues = [''.join(items) for items in
            self._categorie_arr[targets.cpu().numpy()]]
        self.cer_metric.update(preds, trues)
        # print('debug[preds]:', preds, inputs.shape)
        # print('debug[trues]:', trues)

    def compute_metric(self) -> Dict[str, Any]:
        cer = self.cer_metric.compute()
        # wer = self.wer_metric.compute()
        self.cer_metric.reset()
        # self.wer_metric.reset()
        return dict(
            cer=cer,
            # wer=wer,
            c_score=1 - cer,
        )

    def collate_fn(
            self,
            batch: List[Any],
        ) -> Any:

        batch_size = len(batch)
        aligned_width = 0
        aligned_length = 0
        for image, target, _ in batch:
            aligned_width = max(aligned_width, image.shape[2])
            aligned_length = max(aligned_length, target.shape[0])
        aligned_width = math.ceil(aligned_width / 32) * 32
        images = torch.zeros(batch_size, 3, batch[0][0].shape[1], aligned_width)
        targets = torch.zeros(batch_size, aligned_length, dtype=torch.int64)
        input_lengths = torch.zeros(batch_size, dtype=torch.int64)
        target_lengths = torch.zeros(batch_size, dtype=torch.int64)
        for i, (image, target, reverse) in enumerate(batch):
            target_length = target.shape[-1]
            target_lengths[i] = target_length
            targets[i, :target_length] = target
            image_width = image.shape[-1]
            input_lengths[i] = math.ceil(image_width / 8)
            images[i, :, :, :image_width] = image
        return images, targets, input_lengths, target_lengths

    @classmethod
    def get_transforms(
            cls,
            task_name:    str='default',
            max_width:    int=1024,
            align_height: int=32,
        ) -> Callable:

        if task_name == 'default':
            transform_list = [
                transforms.Grayscale(num_output_channels=3),
                RandomInvert(prob=0.5),
                # RandomRotate(4.5, prob=1.), # for test
            ]

        elif task_name == 'natural':
            transform_list= [
                RandomInvert(prob=0.5),
            ]

        elif task_name == 'olhwdb2':
            transform_list = [
                RandomTableLine(prob=0.05),
                InterLine(prob=0.05),
                VerticalLine(prob=0.05),
                RandomLine(prob=0.05),
                RandomPixelTrans(prob=0.5),
                RandomChoices([
                    RandomNoise(prob=0.1),
                    RandomGradient(prob=0.1),
                    RandomBlur(prob=0.1),
                    DropoutVertical(prob=0.1),
                    DropoutHorizontal(prob=0.1),
                ], times=3),
                RandomInvert(prob=0.5),
                transforms.Grayscale(num_output_channels=3),
            ]

        elif task_name == 'printing':
            transform_list = [
                TextDilate(iterations=2, prob=0.1),
                transforms.RandomChoice([
                    NoiseCharacters(prob=0.05),
                    RandomVerticalCrop(prob=0.05),
                ]),
                RandomTableLine(prob=0.05),
                InterLine(prob=0.05),
                transforms.RandomChoice([
                    RandomPad([1, 0.5], prob=0.5),
                    RandomAffine(0.1, prob=0.5),
                    transforms.Compose([
                        transforms.RandomChoice([
                            RandomPad([1, 0.5], prob=0.5),
                            RandomAffine(0.1, prob=0.5),
                        ]),
                        RandomRotate(4.5, prob=1.),
                    ]),
                ]),
                VerticalLine(prob=0.05),
                RandomLine(prob=0.05),
                RandomPixelTrans(prob=0.5),
                RandomChoices([
                    RandomCurve(prob=0.1),
                    RandomNoise(prob=0.1),
                    RandomGradient(prob=0.1),
                    RandomBlur(prob=0.1),
                    RandomLevelScale(prob=0.1),
                    DropoutVertical(prob=0.1),
                    DropoutHorizontal(prob=0.1),
                ], times=3),
                RandomInvert(prob=0.5),
                transforms.Grayscale(num_output_channels=3),
            ]

        elif task_name == 'handwriting':
            transform_list = [
                TextDilate(iterations=2, prob=0.1),
                transforms.RandomChoice([
                    NoiseCharacters(prob=0.05),
                    RandomVerticalCrop(prob=0.05),
                ]),
                RandomTableLine(prob=0.05),
                InterLine(prob=0.05),
                transforms.RandomChoice([
                    RandomPad([1, 0.5], prob=0.5),
                    RandomAffine(0.1, prob=0.5),
                    transforms.Compose([
                        transforms.RandomChoice([
                            RandomPad([1, 0.5], prob=0.5),
                            RandomAffine(0.1, prob=0.5),
                        ]),
                        RandomRotate(4.5, prob=1.),
                    ]),
                ]),
                VerticalLine(prob=0.05),
                RandomLine(prob=0.05),
                RandomPixelTrans(prob=0.5),
                RandomChoices([
                    RandomCurve(prob=0.1),
                    RandomGradient(prob=0.1),
                    RandomLevelScale(prob=0.1),
                    DropoutVertical(prob=0.1),
                    DropoutHorizontal(prob=0.1),
                    transforms.RandomChoice([
                        RandomNoise(prob=0.1),
                        RandomBlur(prob=0.1),
                    ]),
                ], times=3),
                RandomInvert(prob=0.5),
                transforms.Grayscale(num_output_channels=3),
            ]

        else:
            raise ValueError(f'Unsupported the task `{task_name}`')

        transform_list.extend([
            AlignSize(width=max_width, height=align_height),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        return transforms.Compose(transform_list)


class RandomPad:
    def __init__(self, rate:list=[1, 0.5], prob:float=0.5):
        rate = list(rate) if isinstance(rate, Iterable) else [rate]
        self.rate = (rate * 2)[:2]
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        src_width, src_height = img.size
        padding = [int(random.randint(0, int(r * src_height))) for r in self.rate]
        dst_width = padding[0] + src_width
        dst_height = padding[1] + src_height
        new_img = Image.new('L', (dst_width, dst_height), color=0)
        offset = [random.randint(0, p) for p in padding]
        new_img.paste(img, (offset[0], offset[1]))
        return new_img

    def __repr__(self):
        return f'{self.__class__.__name__}(rate={self.rate}, prob={self.prob})'


class RandomRotate:
    def __init__(self, angle:float=2.25, prob:float=0.5):
        assert angle >= 0.

        self.angle = angle
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        w, h = img.size
        v_b = np.array([w / 2, h / 2])
        v_u = np.array([max(0, (w / 2)**2 - h**2)**0.5, h])
        cos_theta = (v_b @ v_u) / (np.linalg.norm(v_b) * np.linalg.norm(v_u))
        theta = np.arccos(cos_theta)
        angle_limit = np.degrees(theta)
        angle = min(self.angle, angle_limit) * random.uniform(-1, 1)
        img = img.rotate(
            angle, resample=Image.Resampling.BILINEAR, expand=1, fillcolor=0)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(angle={self.angle}, prob={self.prob})'


class RandomBlur:
    def __init__(self, radius:int=2, prob:float=0.5):
        self.radius = radius
        self.prob   = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = img.filter(
            ImageFilter.GaussianBlur(random.randint(1, self.radius)))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(radius={self.radius}, prob={self.prob})'


class RandomLine:
    def __init__(self, width:int=2, fill:list=[0, 255], prob:float=0.5):
        self.width = width
        self.fill  = fill
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = img.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)
        draw.line(tuple(random.randint(0, i - 1) for i in (w, h, w, h)),
                  fill=random.randint(self.fill[0], self.fill[1]),
                  width=random.randint(1, self.width))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width}, fill={self.fill}, prob={self.prob})'


class RandomTableLine:
    def __init__(self, width:int=2, bias:int=3, fill:list=[0, 255], prob:float=0.5):
        self.width = width
        self.bias  = bias
        self.fill  = fill
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = img.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)
        anchor = random.randint(0, 3)
        x0 = random.randint(0, self.bias)
        y0 = random.randint(0, self.bias)
        x1 = random.randint(w - self.bias, w) - 1
        y1 = random.randint(h - self.bias, h) - 1
        width = random.randint(1, self.width)
        fill = random.randint(self.fill[0], self.fill[1])
        if anchor == 0:
            draw.line((x0, y0, x1, y0), fill=fill, width=width)
            draw.line((x0, y0, x0, y1), fill=fill, width=width)
        elif anchor == 1:
            draw.line((x1, y0, x0, y0), fill=fill, width=width)
            draw.line((x1, y0, x1, y1), fill=fill, width=width)
        elif anchor == 2:
            draw.line((x1, y1, x0, y1), fill=fill, width=width)
            draw.line((x1, y1, x1, y0), fill=fill, width=width)
        elif anchor == 3:
            draw.line((x0, y1, x1, y1), fill=fill, width=width)
            draw.line((x0, y1, x0, y0), fill=fill, width=width)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width}, bias={self.bias}, fill={self.fill}, prob={self.prob})'


class RandomGradient:
    def __init__(self, prob:float=0.5):
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        w, h = img.size
        T = w
        b = random.randint(0, w)
        x = np.arange(w)
        y = 0.5 * np.sin(x * (2 * np.pi / T) + b) + 0.5
        y = np.clip(y, 0.8, None)
        arr = np.array(img) * y[None, :]
        img = Image.fromarray(arr.astype(np.uint8))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


class RandomInvert:
    def __init__(self, prob:float=0.5):
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = ImageOps.invert(img)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


class RandomLevelScale:
    def __init__(self, rate:list=[0.8, 1.5], prob:float=0.5):
        self.rate = rate
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        width, height = img.size
        width = int(width * random.uniform(self.rate[0], self.rate[1]))
        img = img.resize((width, height), resample=Image.Resampling.BILINEAR)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(rate={self.rate}, prob={self.prob})'


class RandomNoise:
    def __init__(self, density:float=0.3, intensity:int=75, prob:float=0.5):
        self.density   = density
        self.intensity = intensity
        self.prob      = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        shape = img.size[1], img.size[0]
        arr = np.array(img)
        density_mask = np.random.random(shape) < self.density
        noise = np.random.randint(-self.intensity, self.intensity, shape)
        noise = density_mask * noise
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(density={self.density}, intensity={self.intensity}, prob={self.prob})'


class RandomCurve:
    def __init__(self, scale:int=3, prob:float=0.5):
        self.scale = scale
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr = np.asarray(img, dtype=np.uint8)
        new_arr = np.zeros_like(arr)
        bias = self.scale * np.sin(
            np.arange(arr.shape[1]) / arr.shape[0] +
            random.uniform(-math.pi / 2, math.pi / 2))
        for i, bf in enumerate(bias): 
            bi = int(round(bf))
            if bi < 0: 
                height = arr.shape[0] + bi
                new_arr[:height, i] = arr[-bi:, i] 
            else: 
                height = arr.shape[0] - bi 
                new_arr[bi:, i] = arr[:height, i] 
        img = Image.fromarray(new_arr.astype(np.uint8))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale}, prob={self.prob})'


class NoiseCharacters:
    def __init__(self, rate:float=0.2, prob:float=0.5):
        self.rate = rate
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        src_w, src_h = img.size
        dst_w, dst_h = src_w, round(src_h * (1. + self.rate))
        new_img = Image.new('L', (dst_w, dst_h))
        if random.random() < 0.5:
            cropped = img.crop((0, 0, src_w, dst_h - src_h))
            cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            new_img.paste(img, (0, 0))
            new_img.paste(cropped, (0, src_h))
        else:
            cropped = img.crop((0, 2 * src_h - dst_h, src_w, src_h))
            cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            new_img.paste(img, (0, dst_h - src_h))
            new_img.paste(cropped, (0, 0))
        return new_img

    def __repr__(self):
        return f'{self.__class__.__name__}(rate={self.rate}, prob={self.prob})'


class InterLine:
    def __init__(self, width:int=2, beta:float=0.25, fill:list=[0, 255], prob:float=0.5):
        self.width = width
        self.beta  = beta
        self.fill  = fill
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = img.copy()
        w, h = img.size
        y0 = int(random.uniform(self.beta, 1 - self.beta) * h)
        y1 = y0 + random.randint(-3, 3)
        draw = ImageDraw.Draw(img)
        draw.line(
            (0, y0, w, y1),
            fill=random.randint(self.fill[0], self.fill[1]),
            width=random.randint(1, self.width))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width}, beta={self.beta}, fill={self.fill}, prob={self.prob})'


class VerticalLine:
    def __init__(self, width:int=2, beta:float=0.25, fill:list=[0, 255], prob:float=0.5):
        self.width = width
        self.beta  = beta
        self.fill  = fill
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        img = img.copy()
        w, h = img.size
        pd = round(h * self.beta)
        x0 = random.randint(pd, max(w - pd, pd + 1))
        x1 = x0 + random.randint(-3, 3)
        draw = ImageDraw.Draw(img)
        draw.line(
            (x0, 0, x1, h),
            fill=random.randint(self.fill[0], self.fill[1]),
            width=random.randint(1, self.width))
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width}, beta={self.beta}, fill={self.fill}, prob={self.prob})'


class RandomVerticalCrop:
    def __init__(self, rate:float=0.1, prob:float=0.5):
        self.rate = rate
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        x1, y1, x2, y2 = 0, 0, img.size[0], img.size[1]
        random_bias = int(random.uniform(-self.rate, self.rate) * y2)
        if random_bias < 0: y2 += random_bias
        else: y1 += random_bias
        img = img.crop([x1, y1, x2, y2])
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(rate={self.rate}, prob={self.prob})'


class DropoutHorizontal:
    def __init__(self, thresh:float=0.25, prob:float=0.5):
        self.thresh = thresh
        self.prob   = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr  = np.array(img, dtype=np.uint8)
        mask = np.abs(np.random.randn(arr.shape[0])) < self.thresh
        rows = mask.sum()
        if rows == 0: return img
        rate = np.random.random((rows, 1))
        arr[mask] = (arr[mask] * rate).astype(np.uint8)
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(thresh={self.thresh}, prob={self.prob})'


class DropoutVertical:
    def __init__(self, thresh:float=0.25, prob:float=0.5):
        self.thresh = thresh
        self.prob   = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr = np.array(img, dtype=np.uint8)
        mask = np.abs(np.random.randn(arr.shape[1])) < self.thresh
        cols = mask.sum()
        if cols == 0: return img
        rate = np.random.random((1, cols))
        arr[:, mask] = (arr[:, mask] * rate).astype(np.uint8)
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(thresh={self.thresh}, prob={self.prob})'


class RandomPixelTrans:
    def __init__(self, scale:float=0.5, prob:float=0.5):
        self.scale = scale
        self.prob  = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr = np.array(img, dtype=np.uint8)
        scale = random.uniform(self.scale, 1.)
        arr = (arr * scale).astype(np.uint8)
        arr += random.randint(0, int((1 - scale) * 255))
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(scale={self.scale}, prob={self.prob})'


class TextDilate:
    def __init__(self, iterations:int=1, prob:float=0.5):
        self.iterations = iterations
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr = np.array(img, dtype=np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        iterations = random.randint(1, self.iterations)
        arr = cv.dilate(arr, kernel, iterations=iterations)
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(iterations={self.iterations}, prob={self.prob})'


class RandomAffine:
    def __init__(self, factor:float=0.1, prob:float=0.5):
        self.factor = factor
        self.prob = prob

    def __call__(self, img:Image.Image):
        if random.random() >= self.prob: return img
        arr = np.array(img, dtype=np.uint8)
        src_w, src_h = img.size
        src_pts = np.array([
            [0, src_h],
            [0, 0],
            [src_w, 0],
            [src_w, src_h]], dtype=np.float32)
        dst_pts = np.copy(src_pts)
        pull_dist = random.randrange(int(self.factor * src_h), src_h)
        if random.random() > 0.5:
            dst_pts[[0, 3], 0] += pull_dist
        else:
            dst_pts[[1, 2], 0] += pull_dist
        dst_w = src_w + pull_dist
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        arr = cv.warpPerspective(arr, M, (dst_w, src_h), flags=cv.INTER_AREA)
        img = Image.fromarray(arr)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(factor={self.factor}, prob={self.prob})'


class AlignSize:
    def __init__(self, width:int=320, height:int=32):
        self.width = width
        self.height = height

    def __call__(self, img:Image.Image):
        src_width, src_height = img.size
        dst_height = self.height
        dst_width = int(dst_height / src_height * src_width)
        resized = img.resize((min(dst_width, self.width), dst_height), resample=Image.Resampling.BILINEAR)
        return resized

    def __repr__(self):
        return f'{self.__class__.__name__}(width={self.width}, height={self.height})'


class PixelOneZeroScaling:
    def __call__(self, x:torch.Tensor):
        pixel_min = x.min()
        pixel_max = x.max()
        x = (x - pixel_min) / torch.clamp_min(pixel_max - pixel_min, 1e-5)
        return x

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class RandomChoices:
    def __init__(self, transforms:list, times:int):
        self.transforms = transforms
        self.times      = times

    def __call__(self, img:Image.Image):
        indexs = list(range(len(self.transforms)))
        indexs = sorted(random.sample(indexs, k=self.times))
        for i in indexs:
            img = self.transforms[i](img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return f"{format_string}(times={self.times})"
