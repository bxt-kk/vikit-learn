from typing import Any, Callable, Tuple
import os

from PIL import Image
from torchvision.datasets.vision import VisionDataset


class Places365(VisionDataset):
    '''`Places365 <http://places2.csail.mit.edu/index.html>`_ classification dataset.

    Args:
        root: Root directory where images are downloaded to.
        split: The dataset split, supports ``"train"``(default), ``"val"``.
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
        split:            str='train',
        transform:        Callable | None=None,
        target_transform: Callable | None=None,
        transforms:       Callable | None=None,
    ):
        super().__init__(root, transforms, transform, target_transform)
        assert split in ['train', 'val']
        self.dataset_dir = root
        self.classes = sorted(os.listdir(os.path.join(root, split)))
        paths_file = os.path.join(root, f'{split}.txt')
        self.paths = []
        with open(paths_file) as f:
            for path in f:
                path = path.strip()
                if not path: continue
                self.paths.append(path)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path:str) -> Image.Image:
        return Image.open(
            os.path.join(self.dataset_dir, path)).convert('RGB')

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
