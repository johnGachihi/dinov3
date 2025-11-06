"""
Patch Cosine Similarity Visualization for DINOv2/DINOv3

This script loads a DINOv3 model from a config file and optional checkpoint,
loads images from the Sen2Venus dataset, computes the cosine similarity between
a selected patch and all other patches, and visualizes the result as a heatmap overlay.

Usage:
    python patch_similarity.py --config path/to/config.yaml --checkpoint path/to/checkpoint.pth \
        --image_index 0 --dataset_root /path/to/sen2venus --patch_idx 128

Requirements:
    - torch
    - torchvision
    - numpy
    - matplotlib
    - PIL
    - omegaconf
    - h5py
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path


def load_model(config_file, checkpoint_path=None, device='cuda'):
  """
  Load a DINOv3 model from a config file.

  Args:
      config_file: Path to the config YAML file
      checkpoint_path: Path to the model checkpoint (.pth file or directory) (optional)
      device: Device to load the model on ('cuda' or 'cpu')

  Returns:
      model: Loaded model in evaluation mode
  """
  from omegaconf import OmegaConf
  import math

  # Add parent directory to path to import dinov3
  dinov3_path = Path(__file__).parent.parent
  if str(dinov3_path) not in sys.path:
    sys.path.insert(0, str(dinov3_path))

  from dinov3.models import build_model_from_cfg
  from dinov3.checkpointer import load_checkpoint

  # Load config
  print(f"Loading config from {config_file}...")
  cfg = OmegaConf.load(config_file)

  # Build model on meta device (teacher only for inference)
  print("Building model from config...")
  with torch.device("meta"):
    model, embed_dim = build_model_from_cfg(cfg, only_teacher=True)

  # Move model to device and initialize
  print(f"Moving model to {device}...")
  model._apply(
    lambda t: torch.full_like(
      t,
      fill_value=math.nan if t.dtype.is_floating_point else (2 ** (t.dtype.itemsize * 8 - 1)),
      device=device,
    ),
    recurse=True,
  )
  model.init_weights()

  # Load checkpoint if provided
  if checkpoint_path is not None:
    checkpoint_path = Path(checkpoint_path)
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Check if it's a directory (FSDP checkpoint) or a file
    if checkpoint_path.is_dir():
      # FSDP checkpoint - use load_checkpoint
      print("Detected FSDP checkpoint directory")
      load_checkpoint(
        checkpoint_path,
        model=model,
        strict_loading=False,
      )
    else:
      # Regular checkpoint file
      print("Detected regular checkpoint file")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')

      # Handle different checkpoint formats
      if 'teacher' in checkpoint:
        state_dict = checkpoint['teacher']
        # Remove 'backbone.' prefix if present
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
      elif 'model' in checkpoint:
        state_dict = checkpoint['model']
      elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
      else:
        state_dict = checkpoint

      # Load the state dict
      msg = model.load_state_dict(state_dict, strict=False)
      print(f"Checkpoint loaded: {msg}")
  else:
    print("No checkpoint provided, using randomly initialized weights")

  model.eval()
  return model


def load_and_preprocess_image(image_index, dataset_root, cfg, img_size=518, use_hr=True, split='train'):
  """
  Load and preprocess an image from the Sen2Venus dataset.

  Args:
      image_index: Index of the image in the dataset
      dataset_root: Root directory of the Sen2Venus dataset
      cfg: OmegaConf config object containing normalization parameters
      img_size: Size to resize the image (default: 518)
      use_hr: If True, use high-resolution (Venus) image, otherwise use low-resolution (Sentinel-2)
      split: Dataset split to use (default: 'train')

  Returns:
      img_tensor: Preprocessed image tensor
      original_img: Original image as numpy array for visualization (normalized to 0-1 for first 3 channels)
  """
  # Add parent directory to path to import dinov3
  dinov3_path = Path(__file__).parent.parent
  if str(dinov3_path) not in sys.path:
    sys.path.insert(0, str(dinov3_path))

  from dinov3.data.datasets.sen2venus import Sen2Venus

  # Initialize dataset
  dataset = Sen2Venus(root=dataset_root, split=split)

  # Load image data
  venus_img, sentinel2_img = dataset.get_image_data(image_index)

  # Select which image to use
  img_tensor = venus_img if use_hr else sentinel2_img

  # Get normalization parameters from config
  if use_hr:
    mean = cfg.crops.mean_hr if hasattr(cfg.crops, 'mean_hr') else [0.485, 0.456, 0.406, 0.5]
    std = cfg.crops.std_hr if hasattr(cfg.crops, 'std_hr') else [0.229, 0.224, 0.225, 0.5]
  else:
    mean = cfg.crops.mean_lr if hasattr(cfg.crops, 'mean_lr') else [0.485, 0.456, 0.406, 0.5]
    std = cfg.crops.std_lr if hasattr(cfg.crops, 'std_lr') else [0.229, 0.224, 0.225, 0.5]

  # Ensure we have the right number of channels
  n_channels = img_tensor.shape[0]
  if len(mean) != n_channels:
    mean = mean[:n_channels]
    std = std[:n_channels]

  img_tensor = transforms.v2.functional.to_dtype(img_tensor)

  # Resize if needed (C, H, W)
  if img_tensor.shape[1] != img_size or img_tensor.shape[2] != img_size:
    img_tensor = torch.nn.functional.interpolate(
      img_tensor.unsqueeze(0),
      size=(img_size, img_size),
      mode='bilinear',
      align_corners=False
    ).squeeze(0)

  # Create a visualization image (normalize first 3 channels to 0-1 for RGB visualization)
  original_img = img_tensor[:3][[2, 1, 0]].permute(1, 2, 0).numpy()  # Convert to HWC format
  # Normalize to 0-1 for visualization
  original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
  original_img = (original_img * 255).astype(np.uint8)
  original_img = Image.fromarray(original_img)

  # Normalize the full tensor for model input
  mean_tensor = torch.tensor(mean).view(-1, 1, 1)
  std_tensor = torch.tensor(std).view(-1, 1, 1)
  img_tensor = (img_tensor - mean_tensor) / std_tensor

  # Add batch dimension
  img_tensor = img_tensor.unsqueeze(0)

  return img_tensor, original_img


def extract_patch_features(model, img_tensor, device='cuda'):
  """
  Extract patch features from DINOv2 model.

  Args:
      model: DINOv2 model
      img_tensor: Preprocessed image tensor
      device: Device to run inference on

  Returns:
      patch_features: Tensor of shape (num_patches, feature_dim)
      num_patches_h: Number of patches along height
      num_patches_w: Number of patches along width
  """
  img_tensor = img_tensor.to(device)

  with torch.no_grad():
    # Extract features using forward_features
    output = model.forward_features(img_tensor)

    # Get patch tokens (exclude CLS token at position 0)
    # The output contains both CLS token and patch tokens
    if isinstance(output, dict):
      patch_features = output['x_norm_patchtokens']  # Shape: (batch, num_patches, feature_dim)
    else:
      # If output is a tensor, assume first token is CLS
      patch_features = output[:, 1:, :]  # Shape: (batch, num_patches, feature_dim)

  # Remove batch dimension
  patch_features = patch_features.squeeze(0)  # Shape: (num_patches, feature_dim)

  # Calculate grid dimensions
  num_patches = patch_features.shape[0]
  num_patches_side = int(np.sqrt(num_patches))
  num_patches_h = num_patches_side
  num_patches_w = num_patches_side

  return patch_features, num_patches_h, num_patches_w


def compute_cosine_similarity_map(patch_features, reference_patch_idx):
  """
  Compute cosine similarity between a reference patch and all other patches.

  Args:
      patch_features: Tensor of shape (num_patches, feature_dim)
      reference_patch_idx: Index of the reference patch

  Returns:
      similarity_map: Cosine similarity values for all patches (num_patches,)
  """
  # Get the reference patch feature
  reference_feature = patch_features[reference_patch_idx].unsqueeze(0)  # Shape: (1, feature_dim)

  # Compute cosine similarity with all patches
  # Using PyTorch's cosine_similarity function
  similarity_map = F.cosine_similarity(reference_feature, patch_features, dim=1)

  return similarity_map.cpu().numpy()


def visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                             reference_patch_idx, img_size=518, alpha=0.6, save_path=None):
  """
  Visualize the cosine similarity map as a heatmap overlay on the original image.

  Args:
      original_img: Original PIL image
      similarity_map: Similarity values (num_patches,)
      num_patches_h: Number of patches along height
      num_patches_w: Number of patches along width
      reference_patch_idx: Index of the reference patch
      img_size: Size of the processed image
      alpha: Transparency of the heatmap overlay
      save_path: Path to save the visualization (optional)
  """
  # Reshape similarity map to 2D grid
  similarity_grid = similarity_map.reshape(num_patches_h, num_patches_w)

  # Resize original image to match processed size
  original_img_resized = original_img.resize((img_size, img_size))

  # Calculate reference patch position
  ref_row = reference_patch_idx // num_patches_w
  ref_col = reference_patch_idx % num_patches_w

  # Create figure with subplots
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))

  # Plot 1: Original image with reference patch marker
  axes[0].imshow(original_img_resized)
  patch_size = img_size // num_patches_h
  rect_x = ref_col * patch_size
  rect_y = ref_row * patch_size
  rect = plt.Rectangle((rect_x, rect_y), patch_size, patch_size,
                       fill=False, edgecolor='red', linewidth=3)
  axes[0].add_patch(rect)
  axes[0].set_title(f'Original Image\nReference Patch: [{ref_row}, {ref_col}]', fontsize=14)
  axes[0].axis('off')

  # Plot 2: Similarity heatmap only
  im = axes[1].imshow(similarity_grid, cmap='viridis', vmin=-1, vmax=1)
  axes[1].set_title('Cosine Similarity Heatmap', fontsize=14)
  axes[1].axis('off')
  plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

  # Plot 3: Overlay heatmap on original image
  axes[2].imshow(original_img_resized)
  # Upsample similarity grid to image size for smooth overlay
  similarity_upsampled = np.kron(similarity_grid, np.ones((patch_size, patch_size)))
  # Ensure correct size
  similarity_upsampled = similarity_upsampled[:img_size, :img_size]

  im_overlay = axes[2].imshow(similarity_upsampled, cmap='hot', alpha=alpha, vmin=0, vmax=1)
  axes[2].set_title(f'Similarity Overlay (alpha={alpha})', fontsize=14)
  axes[2].axis('off')
  plt.colorbar(im_overlay, ax=axes[2], fraction=0.046, pad=0.04)

  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")

  plt.show()


def interactive_patch_selection(original_img, model, img_tensor, device, img_size=518, alpha=0.6):
  """
  Interactive mode: click on the image to select a patch and compute similarity.

  Args:
      original_img: Original PIL image
      model: DINOv2 model
      img_tensor: Preprocessed image tensor
      device: Device to run inference on
      img_size: Size of the processed image
      alpha: Transparency of the heatmap overlay
  """
  # Extract patch features once
  patch_features, num_patches_h, num_patches_w = extract_patch_features(model, img_tensor, device)
  patch_size = img_size // num_patches_h

  # Resize original image
  original_img_resized = original_img.resize((img_size, img_size))

  # Create interactive figure
  fig, ax = plt.subplots(figsize=(10, 10))
  ax.imshow(original_img_resized)
  ax.set_title('Click on a patch to compute similarity\n(Close window to exit)', fontsize=14)
  ax.axis('off')

  # Overlay grid
  for i in range(num_patches_h + 1):
    ax.axhline(y=i * patch_size, color='white', alpha=0.3, linewidth=0.5)
  for j in range(num_patches_w + 1):
    ax.axvline(x=j * patch_size, color='white', alpha=0.3, linewidth=0.5)

  def onclick(event):
    if event.inaxes != ax:
      return

    # Get clicked position
    x, y = int(event.xdata), int(event.ydata)

    # Convert to patch index
    patch_col = min(x // patch_size, num_patches_w - 1)
    patch_row = min(y // patch_size, num_patches_h - 1)
    patch_idx = patch_row * num_patches_w + patch_col

    print(f"\nSelected patch: [{patch_row}, {patch_col}] (index: {patch_idx})")

    # Compute similarity
    similarity_map = compute_cosine_similarity_map(patch_features, patch_idx)

    # Visualize
    visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                             patch_idx, img_size, alpha)

  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  plt.show()


def main():
  parser = argparse.ArgumentParser(description='Compute and visualize patch cosine similarity for DINOv2/DINOv3')
  parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
  parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to model checkpoint (.pth file). If not provided, uses randomly initialized weights.')
  parser.add_argument('--image_index', type=int, required=True,
                      help='Index of the image in the Sen2Venus dataset')
  parser.add_argument('--dataset_root', type=str, required=True,
                      help='Root directory of the Sen2Venus dataset')
  parser.add_argument('--use_hr', action='store_true', default=True,
                      help='Use high-resolution (Venus) image. Use --no-use_hr for low-resolution (Sentinel-2)')
  parser.add_argument('--no-use_hr', dest='use_hr', action='store_false',
                      help='Use low-resolution (Sentinel-2) image instead of high-resolution')
  parser.add_argument('--split', type=str, default='train',
                      choices=['train', 'val', 'test'],
                      help='Dataset split to use (default: train)')
  parser.add_argument('--patch_idx', type=int, default=None,
                      help='Index of reference patch (0 to num_patches-1). If not provided, enters interactive mode.')
  parser.add_argument('--img_size', type=int, default=518,
                      help='Image size for processing (default: 518)')
  parser.add_argument('--alpha', type=float, default=0.6,
                      help='Transparency of heatmap overlay (0-1, default: 0.6)')
  parser.add_argument('--device', type=str, default='cuda',
                      choices=['cuda', 'cpu'],
                      help='Device to run inference on')
  parser.add_argument('--save_path', type=str, default=None,
                      help='Path to save the visualization (optional)')

  args = parser.parse_args()

  # Check device availability
  if args.device == 'cuda' and not torch.cuda.is_available():
    print("CUDA not available, using CPU instead")
    args.device = 'cpu'

  print(f"Using device: {args.device}")

  # Load config first (needed for preprocessing)
  from omegaconf import OmegaConf
  cfg = OmegaConf.load(args.config)

  # Load model
  model = load_model(args.config, args.checkpoint, args.device)

  # Load and preprocess image
  print(f"Loading image index {args.image_index} from {args.dataset_root}...")
  print(f"Using {'high-resolution (Venus)' if args.use_hr else 'low-resolution (Sentinel-2)'} image")
  img_tensor, original_img = load_and_preprocess_image(
    args.image_index, args.dataset_root, cfg, args.img_size, args.use_hr, args.split
  )

  # Extract patch features
  print("Extracting patch features...")
  patch_features, num_patches_h, num_patches_w = extract_patch_features(model, img_tensor, args.device)
  num_patches = patch_features.shape[0]
  print(f"Extracted {num_patches} patches ({num_patches_h}x{num_patches_w})")

  # Interactive or single patch mode
  if args.patch_idx is None:
    print("\nEntering interactive mode...")
    print("Click on any patch in the image to compute its similarity with all other patches.")
    interactive_patch_selection(original_img, model, img_tensor, args.device, args.img_size, args.alpha)
  else:
    # Validate patch index
    if args.patch_idx < 0 or args.patch_idx >= num_patches:
      print(f"Error: patch_idx must be between 0 and {num_patches - 1}")
      sys.exit(1)

    # Compute similarity map
    print(f"Computing cosine similarity for patch {args.patch_idx}...")
    similarity_map = compute_cosine_similarity_map(patch_features, args.patch_idx)

    # Visualize
    print("Generating visualization...")
    visualize_similarity_map(original_img, similarity_map, num_patches_h, num_patches_w,
                             args.patch_idx, args.img_size, args.alpha, args.save_path)


if __name__ == '__main__':
  main()
