# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image

from .decoders import Decoder, IdentityDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class PASTIS(ExtendedVisionDataset):
    """
    PASTIS Dataset for crop type classification from Sentinel-2 time series.

    Dataset characteristics:
    - Temporal Sentinel-2 data (13 timesteps, 10 bands)
    - Image size: 128x128 (can be tiled to 4x 64x64)
    - Splits: Train (Folds 1-3), Val (Fold 4), Test (Fold 5)
    - 20 crop type classes
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
        return_first_in_time: bool = True,
        use_tiles: bool = True,
    ) -> None:
        """
        Args:
            root: Path to the data directory containing pastis.h5
            split: One of 'train', 'val', 'test'
            transforms: Combined image and target transforms
            transform: Image-only transforms
            target_transform: Target-only transforms
            image_decoder: Decoder for image data
            target_decoder: Decoder for target data
            return_first_in_time: If True, return only first timestep (for spatial-only models)
            use_tiles: If True, split 128x128 images into 4x 64x64 tiles
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=image_decoder,
            target_decoder=target_decoder,
        )

        self.h5_path = os.path.join(root, 'pastis.h5')
        self.norm_stats_path = os.path.join(root, 'PASTIS/NORM_S2_patch.json')
        self.split = split
        self.return_first_in_time = return_first_in_time
        self.use_tiles = use_tiles
        self.tiles_per_img = 4 if use_tiles else 1

        # Load normalization statistics
        with open(self.norm_stats_path, "r") as f:
            self.norm_config = json.load(f)

        # Set up folds for splits
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        if split == "train":
            self.folds_to_use = [1, 2, 3]
        elif split == "val":
            self.folds_to_use = [4]
        elif split == "test":
            self.folds_to_use = [5]

        # Load indices for the selected folds
        with h5py.File(self.h5_path, "r") as f:
            all_folds = f["fold"][:]
            self.indices = np.where(np.isin(all_folds, self.folds_to_use))[0]

        # If using tiles, expand the dataset
        if use_tiles:
            self.active_indices = list(range(len(self.indices) * self.tiles_per_img))
        else:
            self.active_indices = list(range(len(self.indices)))

        self.h5_file = None  # Will be opened lazily per worker

    def __len__(self) -> int:
        return len(self.active_indices)

    def _normalize_bands(self, image: np.ndarray, fold: int) -> np.ndarray:
        """Normalize bands using fold-specific statistics."""
        fold_mapping = {1: "Fold_1", 2: "Fold_2", 3: "Fold_3", 4: "Fold_4", 5: "Fold_5"}
        fold_key = fold_mapping[fold]

        means = np.array(self.norm_config[fold_key]["mean"]).reshape(-1, 1, 1)
        stds = np.array(self.norm_config[fold_key]["std"]).reshape(-1, 1, 1)

        return (image - means) / stds

    def get_image_data(self, index: int) -> torch.Tensor:
        """Load and preprocess image data."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        dataset_idx = self.active_indices[index]

        if self.use_tiles:
            img_idx = dataset_idx // self.tiles_per_img
            tile_idx = dataset_idx % self.tiles_per_img
        else:
            img_idx = dataset_idx
            tile_idx = None

        # Load image and fold information
        images = self.h5_file["sentinel2_ts"][self.indices[img_idx]]  # (13, 10, 128, 128)
        fold = self.h5_file["fold"][self.indices[img_idx]]

        # Normalize each timestep
        normed_images = []
        for t in range(images.shape[0]):
            single_timestep = images[t]
            normed = self._normalize_bands(single_timestep, fold)
            normed_images.append(torch.from_numpy(normed).float())

        normed_images = torch.stack(normed_images)  # (13, 10, 128, 128)

        # Select first timestep if requested
        if self.return_first_in_time:
            normed_images = normed_images[0]  # (10, 128, 128)

        # Extract tile if using tiles
        if self.use_tiles and tile_idx is not None:
            subtiles_per_dim = 2
            pixels_per_dim = 128 // subtiles_per_dim  # 64

            row_idx = tile_idx // subtiles_per_dim
            col_idx = tile_idx % subtiles_per_dim

            if self.return_first_in_time:
                normed_images = normed_images[
                    :,
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                ]
            else:
                normed_images = normed_images[
                    :, :,
                    row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                    col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
                ]

        # Select BGRI bands (indices 1, 2, 3, 6 -> Blue, Green, Red, NIR)
        if self.return_first_in_time:
            normed_images = normed_images[[1, 2, 3, 6]]  # (4, H, W)
        else:
            normed_images = normed_images[:, [1, 2, 3, 6]]  # (13, 4, H, W)

        return normed_images

    def get_target(self, index: int) -> torch.Tensor:
        """Load segmentation mask."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        dataset_idx = self.active_indices[index]

        if self.use_tiles:
            img_idx = dataset_idx // self.tiles_per_img
            tile_idx = dataset_idx % self.tiles_per_img
        else:
            img_idx = dataset_idx
            tile_idx = None

        # Load label
        labels = torch.from_numpy(
            self.h5_file["label"][self.indices[img_idx]]
        ).long().squeeze(0)  # (128, 128)

        # Extract tile if using tiles
        if self.use_tiles and tile_idx is not None:
            subtiles_per_dim = 2
            pixels_per_dim = 128 // subtiles_per_dim  # 64

            row_idx = tile_idx // subtiles_per_dim
            col_idx = tile_idx % subtiles_per_dim

            labels = labels[
                row_idx * pixels_per_dim : (row_idx + 1) * pixels_per_dim,
                col_idx * pixels_per_dim : (col_idx + 1) * pixels_per_dim,
            ]

        return labels