import os

from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


class PlantHealthDataset(Dataset):
    """Images of one cultivation line with their expert health class.

    ``frame`` is a split CSV written by ``scripts/prepare_splits.py`` (columns include
    image_path, plant_id, week, view, health_level and the contiguous ``label``).
    ``__getitem__`` returns (image, label) by default, or (image, label, index) when
    ``return_index`` is set so predictions can be mapped back to plant/week/view.
    """

    def __init__(self, frame, image_root, transform=None, return_index=False):
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.return_index = return_index
        self.targets = self.frame['label'].astype(int).tolist()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        img = pil_loader(os.path.join(self.image_root, row['image_path']))
        if self.transform is not None:
            img = self.transform(img)
        label = int(row['label'])
        if self.return_index:
            return img, label, index
        return img, label
