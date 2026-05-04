import torchvision.transforms as tvs_trans
import torch
from openood.utils.config import Config
from .base_preprocessor import BasePreprocessor
from .transform import Convert


class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, image_size=256):
        super().__init__()
        self.num_crop = num_crop
        self.image_size = image_size
  

    def forward(self, image):
        cropper = tvs_trans.RandomResizedCrop(
            self.image_size,
            scale=(0.95, 1.0)
        )
        # cropper = tvs_trans.CenterCrop(self.image_size)
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"

class MulticropPreProcessor(BasePreprocessor):
    """For test and validation dataset standard image transformation."""
    def __init__(self, config: Config):
        super(MulticropPreProcessor, self).__init__(config)
        self.transform = tvs_trans.Compose([
            Convert('RGB'),
            tvs_trans.Resize(self.pre_size, interpolation=self.interpolation),
            tvs_trans.ToTensor(),
            MultiRandomCrop(
                num_crop=5, image_size=self.image_size
            ),
            tvs_trans.Normalize(mean=self.mean, std=self.std),
        ])

