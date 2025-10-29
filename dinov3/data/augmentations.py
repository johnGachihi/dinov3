# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        teacher_no_color_jitter=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.teacher_no_color_jitter = teacher_no_color_jitter
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"teacher_no_color_jitter: {teacher_no_color_jitter}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"share_color_jitter: {share_color_jitter}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes. We first take a crop of the maximum size
        # and then resize it to the desired size for global and gram teacher crops.
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # random resized crop and flip
        self.geometric_augmentation_global = v2.Compose(
            [
                v2.RandomResizedCrop(
                    global_crop_max_size,
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        resize_global = nn.Identity()  # Resize transform applied to global crops after random crop
        self.resize_global_post_transf = (
            nn.Identity()
        )  # Resize transform applied to global crops after all other transforms
        self.resize_gram_teacher = None  # Resize transform applied to crops for gram teacher
        if gram_teacher_crops_size is not None:
            # All resize transforms will do nothing if the crop size is already the desired size.
            if gram_teacher_no_distortions:
                # When there a no distortions for the gram teacher crop, we can resize before the distortions.
                # This is the preferred order, because it keeps the image size for the augmentations consistent,
                # which matters e.g. for GaussianBlur.
                resize_global = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                # When there a no distortions for the gram teacher crop, we need to resize after the distortions,
                # because the distortions are shared between global and gram teacher crops.
                self.resize_global_post_transf = v2.Resize(
                    global_crops_size,
                    interpolation=v2.InterpolationMode.BICUBIC,
                )

            self.resize_gram_teacher = v2.Resize(
                gram_teacher_crops_size,
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        self.geometric_augmentation_local = v2.Compose(
            [
                v2.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                ),
                v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
            ]
        )

        # color distortions / blurring
        color_jittering = v2.Compose(
            [
                v2.RandomApply(
                    [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                v2.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = v2.Compose(
            [
                GaussianBlur(p=0.1),
                v2.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean, std=std),
            ]
        )

        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([resize_global, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([resize_global, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose(
                [resize_global, color_jittering, global_transfo1_extra, self.normalize]
            )
            self.global_transfo2 = v2.Compose(
                [resize_global, color_jittering, global_transfo2_extra, self.normalize]
            )
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}
        output["weak_flag"] = True  # some residual from mugs

        if self.share_color_jitter:
            image = self.color_jittering(image)

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1_transf = self.global_transfo1(im1_base)
        global_crop_1 = self.resize_global_post_transf(global_crop_1_transf)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2_transf = self.global_transfo2(im2_base)
        global_crop_2 = self.resize_global_post_transf(global_crop_2_transf)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        if self.teacher_no_color_jitter:
            output["global_crops_teacher"] = [
                self.normalize(im1_base),
                self.normalize(im2_base),
            ]
        else:
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        if self.gram_teacher_crops_size is not None:
            # crops for gram teacher:
            if self.gram_teacher_no_distortions:
                gram_crop_1 = self.normalize(self.resize_gram_teacher(im1_base))
                gram_crop_2 = self.normalize(self.resize_gram_teacher(im2_base))
            else:
                gram_crop_1 = self.resize_gram_teacher(global_crop_1_transf)
                gram_crop_2 = self.resize_gram_teacher(global_crop_2_transf)
            output["gram_teacher_crops"] = [gram_crop_1, gram_crop_2]

        # local crops:
        if self.local_crops_subset_of_global_crops:
            _local_crops = [self.local_transfo(im1_base) for _ in range(self.local_crops_number // 2)] + [
                self.local_transfo(im2_base) for _ in range(self.local_crops_number // 2)
            ]

            local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size
            for img in _local_crops:
                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                local_crops.append(img[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops"] = local_crops
            output["offsets"] = offsets
        else:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output


class DataAugmentationDINOHRLR(object):
    """
    DataAugmentation for DINO with HR/LR multispectral satellite image pairs.
    Applies the same geometric transformations to both HR and LR images
    to maintain spatial correspondence.

    Designed for multispectral satellite imagery (>3 bands), so RGB-specific
    augmentations like color jittering and solarization are not applied.
    Only geometric transforms and Gaussian blur are used.
    """

    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        gram_teacher_crops_size=None,
        gram_teacher_no_distortions=False,
        local_crops_subset_of_global_crops=False,
        patch_size=16,
        horizontal_flips=True,
        mean_hr=IMAGENET_DEFAULT_MEAN,
        std_hr=IMAGENET_DEFAULT_STD,
        mean_lr=IMAGENET_DEFAULT_MEAN,
        std_lr=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.gram_teacher_crops_size = gram_teacher_crops_size
        self.gram_teacher_no_distortions = gram_teacher_no_distortions
        self.local_crops_subset_of_global_crops = local_crops_subset_of_global_crops
        self.patch_size = patch_size
        self.mean_hr = mean_hr
        self.std_hr = std_hr
        self.mean_lr = mean_lr
        self.std_lr = std_lr

        logger.info("###################################")
        logger.info("Using HR/LR multispectral data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"gram_crops_size: {gram_teacher_crops_size}")
        logger.info(f"gram_teacher_no_distortions: {gram_teacher_no_distortions}")
        logger.info(f"local_crops_subset_of_global_crops: {local_crops_subset_of_global_crops}")
        logger.info(f"patch_size if local_crops_subset_of_global_crops: {patch_size}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info(f"mean_hr: {mean_hr}")
        logger.info(f"std_hr: {std_hr}")
        logger.info(f"mean_lr: {mean_lr}")
        logger.info(f"std_lr: {std_lr}")
        logger.info("###################################")

        # Global crops and gram teacher crops can have different sizes
        global_crop_max_size = max(global_crops_size, gram_teacher_crops_size if gram_teacher_crops_size else 0)

        # Geometric augmentation for global crops
        # We'll use RandomResizedCrop but need to apply same crop to both HR and LR
        self.global_crops_scale = global_crops_scale
        self.global_crop_max_size = global_crop_max_size
        self.horizontal_flip_p = 0.5 if horizontal_flips else 0.0

        # Resize transforms
        resize_global = nn.Identity()
        self.resize_global_post_transf = nn.Identity()
        self.resize_gram_teacher = None

        if gram_teacher_crops_size is not None:
            if gram_teacher_no_distortions:
                resize_global = v2.Resize(
                    [global_crops_size, global_crops_size],
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            else:
                self.resize_global_post_transf = v2.Resize(
                    [global_crops_size, global_crops_size],
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            self.resize_gram_teacher = v2.Resize(
                [gram_teacher_crops_size, gram_teacher_crops_size],
                interpolation=v2.InterpolationMode.BICUBIC,
            )

        # Blurring (no color distortions for multispectral data)
        self.global_blur1 = GaussianBlur(p=1.0)
        self.global_blur2 = GaussianBlur(p=0.1)
        self.local_blur = GaussianBlur(p=0.5)

        # Normalization - separate for HR and LR
        self.normalize_hr = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean_hr, std=std_hr),
            ]
        )

        self.normalize_lr = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                make_normalize_transform(mean=mean_lr, std=std_lr),
            ]
        )

        # Resize transform
        self.resize_global = resize_global

    def _apply_random_resized_crop(self, hr_image, lr_image, size, scale):
        """
        Apply the same random resized crop to both HR and LR images.
        Returns cropped HR and LR images with the same crop parameters.
        """
        # Get random crop parameters
        from torchvision.transforms import RandomResizedCrop

        # Use HR image to determine crop parameters
        i, j, h, w = RandomResizedCrop.get_params(hr_image, scale=scale, ratio=(3./4., 4./3.))

        # Apply same crop to HR
        hr_cropped = v2.functional.resized_crop(
            hr_image, i, j, h, w, [size, size],
            interpolation=v2.InterpolationMode.BICUBIC
        )

        # Calculate corresponding coordinates for LR image
        hr_height, hr_width = hr_image.shape[-2:]
        lr_height, lr_width = lr_image.shape[-2:]

        scale_h = lr_height / hr_height
        scale_w = lr_width / hr_width

        lr_i = int(i * scale_h)
        lr_j = int(j * scale_w)
        lr_h = int(h * scale_h)
        lr_w = int(w * scale_w)

        # Apply corresponding crop to LR
        lr_cropped = v2.functional.resized_crop(
            lr_image, lr_i, lr_j, lr_h, lr_w, [size, size],
            interpolation=v2.InterpolationMode.BICUBIC
        )

        return hr_cropped, lr_cropped

    def _apply_horizontal_flip(self, hr_image, lr_image):
        """Apply the same horizontal flip to both HR and LR images."""
        if torch.rand(1) < self.horizontal_flip_p:
            hr_image = v2.functional.hflip(hr_image)
            lr_image = v2.functional.hflip(lr_image)
        return hr_image, lr_image

    def _apply_synchronized_transform(self, hr_image, lr_image, transform):
        """
        Apply the same random transform to both HR and LR images.
        Saves and restores random state to ensure synchronized randomness.
        """
        # Save current random state
        rng_state = torch.get_rng_state()

        # Apply transform to HR
        hr_transformed = transform(hr_image)

        # Restore random state and apply same transform to LR
        torch.set_rng_state(rng_state)
        lr_transformed = transform(lr_image)

        return hr_transformed, lr_transformed

    def __call__(self, hrlr_image):
        """
        Args:
            hrlr_image: (hr_image, lr_image)

        Returns:
            Dictionary with HR and LR crops for global and local views
        """
        hr_image, lr_image = hrlr_image

        output = {}
        output["weak_flag"] = True

        # Global crop 1 - Apply synchronized blur to both HR and LR
        hr_im1_base, lr_im1_base = self._apply_random_resized_crop(
            hr_image, lr_image, self.global_crop_max_size, self.global_crops_scale
        )
        hr_im1_base, lr_im1_base = self._apply_horizontal_flip(hr_im1_base, lr_im1_base)

        # Resize
        hr_im1_resized = self.resize_global(hr_im1_base)
        lr_im1_resized = self.resize_global(lr_im1_base)

        # Apply synchronized blur
        hr_im1_blurred, lr_im1_blurred = self._apply_synchronized_transform(
            hr_im1_resized, lr_im1_resized, self.global_blur1
        )

        # Normalize
        hr_global_crop_1_transf = self.normalize_hr(hr_im1_blurred)
        lr_global_crop_1_transf = self.normalize_lr(lr_im1_blurred)

        hr_global_crop_1 = self.resize_global_post_transf(hr_global_crop_1_transf)
        lr_global_crop_1 = self.resize_global_post_transf(lr_global_crop_1_transf)

        # Global crop 2 - Apply synchronized blur to both HR and LR
        hr_im2_base, lr_im2_base = self._apply_random_resized_crop(
            hr_image, lr_image, self.global_crop_max_size, self.global_crops_scale
        )
        hr_im2_base, lr_im2_base = self._apply_horizontal_flip(hr_im2_base, lr_im2_base)

        # Resize
        hr_im2_resized = self.resize_global(hr_im2_base)
        lr_im2_resized = self.resize_global(lr_im2_base)

        # Apply synchronized blur
        hr_im2_blurred, lr_im2_blurred = self._apply_synchronized_transform(
            hr_im2_resized, lr_im2_resized, self.global_blur2
        )

        # Normalize
        hr_global_crop_2_transf = self.normalize_hr(hr_im2_blurred)
        lr_global_crop_2_transf = self.normalize_lr(lr_im2_blurred)

        hr_global_crop_2 = self.resize_global_post_transf(hr_global_crop_2_transf)
        lr_global_crop_2 = self.resize_global_post_transf(lr_global_crop_2_transf)

        output["global_crops_hr"] = [hr_global_crop_1, hr_global_crop_2]
        output["global_crops_lr"] = [lr_global_crop_1, lr_global_crop_2]

        # Global crops for teacher
        output["global_crops_teacher_hr"] = [hr_global_crop_1, hr_global_crop_2]
        output["global_crops_teacher_lr"] = [lr_global_crop_1, lr_global_crop_2]

        # Gram teacher crops
        if self.gram_teacher_crops_size is not None:
            if self.gram_teacher_no_distortions:
                hr_gram_crop_1 = self.normalize_hr(self.resize_gram_teacher(hr_im1_base))
                hr_gram_crop_2 = self.normalize_hr(self.resize_gram_teacher(hr_im2_base))
                lr_gram_crop_1 = self.normalize_lr(self.resize_gram_teacher(lr_im1_base))
                lr_gram_crop_2 = self.normalize_lr(self.resize_gram_teacher(lr_im2_base))
            else:
                hr_gram_crop_1 = self.resize_gram_teacher(hr_global_crop_1_transf)
                hr_gram_crop_2 = self.resize_gram_teacher(hr_global_crop_2_transf)
                lr_gram_crop_1 = self.resize_gram_teacher(lr_global_crop_1_transf)
                lr_gram_crop_2 = self.resize_gram_teacher(lr_global_crop_2_transf)

            output["gram_teacher_crops_hr"] = [hr_gram_crop_1, hr_gram_crop_2]
            output["gram_teacher_crops_lr"] = [lr_gram_crop_1, lr_gram_crop_2]

        # Local crops
        if self.local_crops_subset_of_global_crops:
            # Local crops as subsets of global crops - need synchronized blur
            hr_local_crops = []
            lr_local_crops = []
            offsets = []
            gs = self.global_crops_size
            ls = self.local_crops_size

            for _ in range(self.local_crops_number // 2):
                # From im1_base
                hr_blurred, lr_blurred = self._apply_synchronized_transform(
                    hr_im1_base, lr_im1_base, self.local_blur
                )
                hr_norm = self.normalize_hr(hr_blurred)
                lr_norm = self.normalize_lr(lr_blurred)

                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                hr_local_crops.append(hr_norm[:, rx : rx + ls, ry : ry + ls])
                lr_local_crops.append(lr_norm[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            for _ in range(self.local_crops_number // 2):
                # From im2_base
                hr_blurred, lr_blurred = self._apply_synchronized_transform(
                    hr_im2_base, lr_im2_base, self.local_blur
                )
                hr_norm = self.normalize_hr(hr_blurred)
                lr_norm = self.normalize_lr(lr_blurred)

                rx, ry = np.random.randint(0, (gs - ls) // self.patch_size, 2) * self.patch_size
                hr_local_crops.append(hr_norm[:, rx : rx + ls, ry : ry + ls])
                lr_local_crops.append(lr_norm[:, rx : rx + ls, ry : ry + ls])
                offsets.append((rx, ry))

            output["local_crops_hr"] = hr_local_crops
            output["local_crops_lr"] = lr_local_crops
            output["offsets"] = offsets
        else:
            # Independent local crops
            hr_local_crops = []
            lr_local_crops = []

            for _ in range(self.local_crops_number):
                hr_crop, lr_crop = self._apply_random_resized_crop(
                    hr_image, lr_image, self.local_crops_size, self.local_crops_scale
                )
                hr_crop, lr_crop = self._apply_horizontal_flip(hr_crop, lr_crop)

                # Apply synchronized blur
                hr_blurred, lr_blurred = self._apply_synchronized_transform(
                    hr_crop, lr_crop, self.local_blur
                )

                hr_local_crops.append(self.normalize_hr(hr_blurred))
                lr_local_crops.append(self.normalize_lr(lr_blurred))

            output["local_crops_hr"] = hr_local_crops
            output["local_crops_lr"] = lr_local_crops
            output["offsets"] = ()

        return output
