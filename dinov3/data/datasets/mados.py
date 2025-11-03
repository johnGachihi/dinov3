# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import os
from typing import Callable, Optional

import h5py
import numpy as np
import torch

from .decoders import Decoder, IdentityDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class MADOS(ExtendedVisionDataset):
    """
    MADOS Dataset for land cover classification from Sentinel-2.

    Dataset characteristics:
    - Single-time Sentinel-2 data (10 bands)
    - Image size: 80x80
    - Splits: Train, Val, Test (defined in HDF5)
    - 15 land cover classes (labels 1-15 mapped to 0-14)
    - Handles NaN pixels by marking as ignore_index
    - Uses BGRI bands (Blue, Green, Red, NIR - indices [1,2,3,6])
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder: Decoder = IdentityDecoder,
        target_decoder: Decoder = TargetDecoder,
    ) -> None:
        """
        Args:
            root: Path to the data directory containing mados.h5
            split: One of 'train', 'val', 'test'
            transforms: Combined image and target transforms
            transform: Image-only transforms
            target_transform: Target-only transforms
            image_decoder: Decoder for image data
            target_decoder: Decoder for target data
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        self.h5_path = os.path.join(root, 'mados.h5')
        self.norm_stats_path = os.path.join(root, 'NORM_CONFIG.json')
        self.split = split

        # Load normalization statistics
        with open(self.norm_stats_path, "r") as f:
            self.norm_config = json.load(f)

        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        # Load indices for the selected split
        with h5py.File(self.h5_path, "r") as f:
            all_splits = f["split"][:]
            self.indices = np.where(np.isin(all_splits, [split.encode('utf-8')]))[0]

        self.h5_file = None  # Will be opened lazily per worker

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize_bands(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize bands using global statistics."""
        means = torch.tensor(self.norm_config["mean"]).reshape(-1, 1, 1)
        stds = torch.tensor(self.norm_config["std"]).reshape(-1, 1, 1)
        return (image - means) / stds

    def get_image_data(self, index: int) -> torch.Tensor:
        """Load and preprocess image data."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Load image (10, 80, 80)
        image = torch.from_numpy(self.h5_file["images"][self.indices[index]]).float()

        # Fill NaN pixels with 0 (will be masked out in labels)
        image = torch.where(torch.isnan(image), 0.0, image)

        # Normalize
        image = self._normalize_bands(image)

        # Select BGRI bands (indices 1, 2, 3, 6 -> Blue, Green, Red, NIR)
        image = image[[1, 2, 3, 6]]  # (4, 80, 80)

        return image

    def get_target(self, index: int) -> torch.Tensor:
        """Load segmentation mask and handle special cases."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        # Load label (80, 80)
        label = torch.from_numpy(self.h5_file["label"][self.indices[index]]).long()

        # Load image to detect NaN pixels
        image = torch.from_numpy(self.h5_file["images"][self.indices[index]]).float()
        nan_mask = torch.isnan(image).any(dim=0)

        # Mark NaN pixels as ignore_index
        label = torch.where(nan_mask, 255, label)

        # Set 0 (no-data labels) to -1 (ignored index)
        label = torch.where(label == 0, 255, label)

        # Shift labels 1-15 to 0-14 (only for non-ignored labels)
        label = torch.where((label > 0) & (label < 255), label - 1, label)

        # Add channel dim
        label = label.unsqueeze(0)

        return label
