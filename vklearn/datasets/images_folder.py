from typing import Any, Callable, Tuple
import os
from glob import glob

from PIL import Image
from torchvision.datasets.vision import VisionDataset


class ImagesFolder(VisionDataset):
    '''Images folder classification dataset.

    Args:
        root: Root directory.
        split: The dataset split, supports `"train"`(default), `"val"`.
        extensions: The tuple of extensions. E.g, `("jpg", "jpeg", "png")`
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. E.g, `transforms.PILToTensor`
        target_transform: A function/transform that takes in the
            target and transforms it.
        transforms: A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    '''

    def __init__(
            self,
            root:             str,
            split:            str='train',
            extensions:       Tuple[str]=('jpg', 'jpeg', 'png'),
            transform:        Callable | None=None,
            target_transform: Callable | None=None,
            transforms:       Callable | None=None,
        ):

        super().__init__(root, transforms, transform, target_transform)
        assert split in ['train', 'val', '']
        self.dataset_dir = os.path.join(root, split)
        self.classes = sorted(os.listdir(self.dataset_dir))
        self.paths = []
        for ext in extensions:
            pattern = os.path.join(self.dataset_dir, '*', f'*.{ext}')
            self.paths.extend(sorted(glob(pattern)))

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path:str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _load_label(self, path:str) -> int:
        category = os.path.basename(os.path.dirname(path))
        return self.classes.index(category)

    def __getitem__(self, index:int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(f'Index must be of type integer, got {type(index)} instead.')

        path = self.paths[index]
        image = self._load_image(path)
        target = self._load_label(path)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
