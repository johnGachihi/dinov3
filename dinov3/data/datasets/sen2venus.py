import json
from typing import Optional, Callable, Any

from pathlib import Path

import h5py
import torch

from dinov3.data.datasets.decoders import Decoder, IdentityDecoder, TargetDecoder
from dinov3.data.datasets.extended import ExtendedVisionDataset


class Sen2Venus(ExtendedVisionDataset):
  def __init__(
      self,
      root,
      split="train",
      transforms: Optional[Callable] = None,
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      image_decoder: Decoder = IdentityDecoder,
      target_decoder: Decoder = TargetDecoder,
  ):
    super().__init__(
      root=root,
      transforms=transforms,
      transform=transform,
      target_transform=target_transform,
      image_decoder=image_decoder,
      target_decoder=target_decoder,
    )

    self.hdf5_file = Path(root) / "sen2venus.hdf5"
    splits_file = Path(root) / "splits_v1.json"
    with open(splits_file, "r") as f:
      self.indices = json.load(f)[split]

  def __len__(self):
    return len(self.indices)

  def get_image_data(self, idx):
    with h5py.File(self.hdf5_file, "r") as data_full:
      venus_img = torch.from_numpy(data_full["venus"][self.indices[idx]])
      sentinel2_img = torch.from_numpy(data_full["sentinel2"][self.indices[idx]])

    return (venus_img, sentinel2_img)

  def get_target(self, index: int) -> Any:
    return None