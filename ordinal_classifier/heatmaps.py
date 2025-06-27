"""Heatmap generation functionality using fastai v2."""

import os
import shutil
from pathlib import Path
from typing import Union, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from fastai.vision.all import *

from .core import ShotTypeClassifier


class HeatmapGenerator:
    """Generate activation heatmaps for shot type predictions."""
    
    def __init__(self, classifier: ShotTypeClassifier):
        """Initialize with a trained classifier.
        
        Args:
            classifier: Trained ShotTypeClassifier instance
        """
        self.classifier = classifier
        self.learn = classifier.learn
        
        if self.learn is None:
            raise ValueError("Classifier must have a loaded model")
    
    def _hooked_backward(self, model, xb, target_class: int):
        """Perform backward pass with hooks to capture gradients."""
        # Get the backbone (first part of network, not FC layer)
        backbone = model[0]
        
        with hook_output(backbone) as hook_a:
            with hook_output(backbone, grad=True) as hook_g:
                preds = model(xb)
                preds[0, target_class].backward()
        
        return hook_a, hook_g
    
    def _show_heatmap(self, hm, output_path, orig_img, alpha=0.5, only_heatmap=False):
        """Save heatmap visualization."""
        _, ax = plt.subplots(figsize=(5, 3))
        
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        
        if not only_heatmap:
            orig_img.show(ax)
            # Resize heatmap to match original image dimensions
            img_h, img_w = orig_img.size[1], orig_img.size[0]  # PIL Image size is (width, height)
            hm_resized = torch.nn.functional.interpolate(
                hm.unsqueeze(0).unsqueeze(0), 
                size=(img_h, img_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze().cpu().numpy()
            ax.imshow(hm_resized, alpha=alpha, extent=(0, img_w, img_h, 0),
                      interpolation='spline16', cmap='YlOrRd')
        else:
            ax.imshow(hm.cpu().numpy(), alpha=alpha, interpolation='spline16', cmap='YlOrRd')
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        plt.close('all')
    
    def _save_original(self, img, output_path):
        """Save original image."""
        img.show(figsize=(5, 3))
        
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        plt.close('all')
        
    def generate_single_heatmap(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        alpha: float = 0.5,
        target_class: Optional[str] = None
    ) -> str:
        """Generate heatmap for a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path for output heatmap
            alpha: Blending alpha
            target_class: Target class name (if None, uses predicted class)
            
        Returns:
            Name of predicted/target class
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        img = PILImage.create(image_path)
        
        # Apply transforms manually
        item_tfm = Resize(375)
        batch_tfm = Normalize.from_stats(*imagenet_stats)
        
        # Process image
        x_tfm = item_tfm(img)  # Apply resize
        x_tensor = tensor(x_tfm).float() / 255.0  # Convert to tensor and normalize to [0,1]
        x_tensor = x_tensor.permute(2, 0, 1)  # Change from HWC to CHW
        
        # Create batch and apply normalization
        xb = x_tensor.unsqueeze(0)  # Add batch dimension
        xb = batch_tfm(xb)  # Apply normalization
        
        # Move to same device as model
        model = self.learn.model.eval()
        device = next(model.parameters()).device
        xb = xb.to(device)
        
        # Get prediction
        with torch.no_grad():
            pred = model(xb)
            pred_class_idx = pred.argmax(dim=1).item()
        
        # Determine target class
        if target_class is None:
            target_class_idx = pred_class_idx
            # Get class name from vocabulary
            if hasattr(self.learn.dls, 'vocab'):
                vocab = list(self.learn.dls.vocab)
                if 0 <= pred_class_idx < len(vocab):
                    target_class_name = vocab[pred_class_idx]
                else:
                    target_class_name = f"class_{pred_class_idx}"
            else:
                target_class_name = f"class_{pred_class_idx}"
        else:
            target_class_name = target_class
            # Find target class index in vocabulary
            if hasattr(self.learn.dls, 'vocab'):
                vocab = list(self.learn.dls.vocab)
                try:
                    target_class_idx = vocab.index(target_class)
                except ValueError:
                    target_class_idx = pred_class_idx  # Fallback to predicted class
            else:
                target_class_idx = pred_class_idx
        
        # Generate heatmap
        xb.requires_grad_(True)
        hook_a, hook_g = self._hooked_backward(model, xb, target_class_idx)
        acts = hook_a.stored[0].cpu()
        avg_acts = acts.mean(0)
        
        # Save heatmap
        self._show_heatmap(avg_acts, output_path, img, alpha)
        
        return target_class_name
        
    def generate_heatmaps(
        self,
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        alpha: float = 0.5,
        only_heatmap: bool = False,
        save_original: bool = True
    ) -> None:
        """Generate heatmaps for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save heatmaps
            alpha: Blending alpha for heatmap overlay
            only_heatmap: If True, save only heatmaps without original images
            save_original: If True, also save original images
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Get image files
        files = [f for f in os.listdir(image_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not files:
            print(f"No image files found in {image_dir}")
            return
        
        # Create temporary directory structure for processing
        temp_dir = image_dir / 'temp_processing'
        temp_dir.mkdir(exist_ok=True)
        train_dir = temp_dir / 'train'
        train_dir.mkdir(exist_ok=True)
        img_dir = train_dir / 'img'
        img_dir.mkdir(exist_ok=True)
        
        try:
            # Move files to temp directory
            for file in files:
                shutil.move(image_dir / file, img_dir / file)
            
            # Setup transforms
            item_tfm = Resize(375)
            batch_tfm = Normalize.from_stats(*imagenet_stats)
            
            # Get model
            model = self.learn.model.eval()
            device = next(model.parameters()).device
            
            print(f"Generating heatmaps for {len(files)} images...")
            
            # Process each image
            for idx, file in enumerate(files):
                print(f"Processing {idx+1}/{len(files)}: {file}")
                
                img_path = img_dir / file
                fname_base = Path(file).stem
                
                # Load and process image
                img = PILImage.create(img_path)
                
                # Apply transforms
                x_tfm = item_tfm(img)
                x_tensor = tensor(x_tfm).float() / 255.0
                x_tensor = x_tensor.permute(2, 0, 1)
                
                # Create batch
                xb = x_tensor.unsqueeze(0)
                xb = batch_tfm(xb)
                xb = xb.to(device)
                
                # Get prediction
                with torch.no_grad():
                    pred = model(xb)
                    pred_class = pred.argmax(dim=1).item()
                
                # Generate heatmap
                xb.requires_grad_(True)
                hook_a, hook_g = self._hooked_backward(model, xb, pred_class)
                acts = hook_a.stored[0].cpu()
                avg_acts = acts.mean(0)
                
                # Save original if requested
                if save_original:
                    original_path = output_dir / f"{fname_base}_original.png"
                    self._save_original(img, original_path)
                
                # Save heatmap
                heatmap_path = output_dir / f"{fname_base}_heatmap.png"
                self._show_heatmap(avg_acts, heatmap_path, img, alpha, only_heatmap)
                
        finally:
            # Move files back and cleanup
            for file in files:
                if (img_dir / file).exists():
                    shutil.move(img_dir / file, image_dir / file)
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        print(f"Heatmaps saved to: {output_dir}") 