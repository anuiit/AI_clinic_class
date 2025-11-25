import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    Visualizes which regions of an image the model focuses on for predictions.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The neural network model
            target_layer: The convolutional layer to compute Grad-CAM for
                         (typically the last conv layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = []
        self.activations = []
        
        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Captures the forward pass activations."""
        self.activations.append(output.detach())
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Captures the backward pass gradients."""
        self.gradients.append(grad_output[0].detach())
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_label_idx: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given input.
        
        Args:
            input_tensor: Preprocessed input image tensor [1, C, H, W]
            target_label_idx: Index of target label. If None, uses highest prediction.
            
        Returns:
            cam: Normalized CAM heatmap [H, W] in range [0, 1]
        """
        self.model.eval()
        self.gradients.clear()
        self.activations.clear()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_label_idx is None:
            # For multilabel, get the label with highest probability
            probs = torch.sigmoid(output)
            target_label_idx = probs.argmax().item()
        
        # Backward pass for target label
        self.model.zero_grad()
        output[0, target_label_idx].backward()
        
        # Get gradients and activations
        grads = self.gradients[0].cpu().numpy()  # [1, C, H, W]
        acts = self.activations[0].cpu().numpy()  # [1, C, H, W]
        
        # Global average pooling of gradients
        weights = np.mean(grads, axis=(2, 3))  # [1, C]
        
        # Weighted combination of activation maps
        cam = np.zeros(acts.shape[2:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights[0]):
            cam += w * acts[0, i, :, :]
        
        # Apply ReLU to keep only positive influences
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam
    
    def __del__(self):
        """Remove hooks when object is destroyed."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def get_target_layer(model: torch.nn.Module, model_name: str = None) -> torch.nn.Module:
    """
    Automatically detect the appropriate target layer for Grad-CAM based on model architecture.
    
    Args:
        model: The model (can be wrapped in MultiLabelModel)
        model_name: Optional model name hint
        
    Returns:
        target_layer: The last convolutional layer suitable for Grad-CAM
    """
    # Handle wrapped models (MultiLabelModel, torch.compile, etc.)
    base = model
    if hasattr(model, 'base_model'):
        base = model.base_model
    if hasattr(base, '_orig_mod'):  # torch.compile wrapper
        base = base._orig_mod
    
    # Try to infer model type from attributes or name
    if model_name is None and hasattr(model, 'base_model_name'):
        model_name = model.base_model_name.lower()
    elif model_name:
        model_name = model_name.lower()
    else:
        model_name = ""
    
    # Debug: print model type
    print(f"[GradCAM] Detecting layer for model type: {type(base).__name__}, name hint: '{model_name}'")
    
    # ResNet architectures
    if hasattr(base, 'layer4'):
        print(f"[GradCAM] Detected ResNet architecture")
        return base.layer4[-1]
    
    # EfficientNet architectures
    elif hasattr(base, 'features') and 'efficientnet' in model_name:
        print(f"[GradCAM] Detected EfficientNet architecture")
        features = base.features
        # Find the last convolutional block
        for i in range(len(features) - 1, -1, -1):
            if hasattr(features[i], 'conv'):
                return features[i]
        # Fallback to last feature layer
        return features[-1]
    
    # VGG architectures
    elif hasattr(base, 'features') and 'vgg' in model_name:
        print(f"[GradCAM] Detected VGG architecture")
        return base.features[-1]
    
    # Generic feature extractor (fallback for EfficientNet if name not provided)
    elif hasattr(base, 'features'):
        print(f"[GradCAM] Detected generic 'features' architecture, using last conv block")
        features = base.features
        # Find the last convolutional block
        for i in range(len(features) - 1, -1, -1):
            if hasattr(features[i], 'conv') or hasattr(features[i], 'block'):
                return features[i]
        # Fallback to last feature layer
        return features[-1]
    
    # Vision Transformer
    elif hasattr(base, 'encoder'):
        print(f"[GradCAM] Detected Vision Transformer architecture")
        # ViT doesn't have conv layers, use last encoder layer
        return base.encoder.layers[-1]
    
    else:
        available_attrs = [attr for attr in dir(base) if not attr.startswith('_')][:20]
        raise ValueError(
            f"Could not automatically detect target layer for model type: {type(base).__name__}. "
            f"Available attributes: {available_attrs}"
        )


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image: Original PIL Image
        heatmap: Normalized heatmap array [H, W] in range [0, 1]
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap to use
        
    Returns:
        Superimposed image as numpy array [H, W, 3] in range [0, 255]
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
    
    # Convert heatmap to RGB colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert PIL image to numpy
    img_array = np.array(image)
    
    # Overlay
    superimposed = heatmap_colored * alpha + img_array * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    return superimposed


def visualize_gradcam_for_predictions(
    model: torch.nn.Module,
    dataset,
    label_columns: List[str],
    device: torch.device,
    save_dir: Path,
    num_samples: int = 10,
    threshold: float = 0.5,
    target_layer_name: str = None
):
    """
    Generate Grad-CAM visualizations for model predictions on sample images.
    Shows heatmaps for top predicted labels.
    
    Args:
        model: Trained model
        dataset: Dataset to sample from
        label_columns: List of label names
        device: Device to run on
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        threshold: Prediction threshold
        target_layer_name: Name of target layer (deprecated, auto-detected if None)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Automatically detect target layer
    try:
        target_layer = get_target_layer(model)
        print(f"Using target layer: {target_layer.__class__.__name__}")
    except Exception as e:
        print(f"Warning: Could not auto-detect target layer: {e}")
        if target_layer_name:
            print(f"Falling back to manual layer: {target_layer_name}")
            base = model.base_model if hasattr(model, 'base_model') else model
            target_layer = getattr(base, target_layer_name)
        else:
            raise
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Randomly select samples with time-based seed
    import time
    rng = np.random.default_rng(seed=int(time.time() * 1000) % (2**32))
    indices = rng.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx, sample_idx in enumerate(indices):
        image_tensor, true_labels = dataset[sample_idx]
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).cpu().squeeze().numpy()
        
        pred_labels = (probs >= threshold).astype(int)
        
        # Get top 5 predicted labels by probability
        top_indices = np.argsort(probs)[::-1][:5]
        
        # Denormalize image for display
        img_display = image_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        img_pil = Image.fromarray((img_display * 255).astype(np.uint8))
        
        # Create visualization for this sample
        num_cams = len(top_indices)
        fig, axes = plt.subplots(2, num_cams + 1, figsize=(4 * (num_cams + 1), 8))
        
        # First column: original image and labels
        axes[0, 0].imshow(img_display)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Original Image', fontweight='bold')
        
        # True labels
        true_label_names = [label_columns[i] for i, val in enumerate(true_labels) if val == 1]
        label_text = "TRUE LABELS:\n" + "\n".join([f"• {name}" for name in true_label_names[:10]])
        if len(true_label_names) > 10:
            label_text += f"\n... +{len(true_label_names) - 10}"
        
        axes[1, 0].axis('off')
        axes[1, 0].text(0.1, 0.5, label_text, transform=axes[1, 0].transAxes,
                       fontsize=9, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Generate Grad-CAM for each top label
        for i, label_idx in enumerate(top_indices):
            label_name = label_columns[label_idx]
            prob = probs[label_idx]
            is_true = true_labels[label_idx].item() == 1
            is_pred = pred_labels[label_idx] == 1
            
            # Generate CAM for this label
            cam = gradcam.generate_cam(
                image_tensor.unsqueeze(0).to(device),
                target_label_idx=label_idx
            )
            
            # Overlay on image
            superimposed = overlay_heatmap_on_image(img_pil, cam, alpha=0.4)
            
            # Top row: heatmap overlay
            axes[0, i + 1].imshow(superimposed)
            axes[0, i + 1].axis('off')
            
            # Title with label info
            status = ""
            if is_true and is_pred:
                status = "✓ TP"
                color = 'green'
            elif is_pred and not is_true:
                status = "✗ FP"
                color = 'orange'
            elif is_true and not is_pred:
                status = "✗ FN"
                color = 'red'
            else:
                status = "TN"
                color = 'gray'
            
            axes[0, i + 1].set_title(
                f"{label_name}\nProb: {prob:.3f} {status}",
                fontsize=9,
                color=color,
                fontweight='bold'
            )
            
            # Bottom row: heatmap only
            axes[1, i + 1].imshow(cam, cmap='jet')
            axes[1, i + 1].axis('off')
            axes[1, i + 1].set_title('Activation Map', fontsize=8)
        
        plt.suptitle(f'Grad-CAM Visualization - Sample {idx + 1}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f'gradcam_sample_{idx + 1}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated Grad-CAM for sample {idx + 1}/{len(indices)}")


def visualize_gradcam_for_specific_labels(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    label_indices: List[int],
    label_columns: List[str],
    device: torch.device,
    save_path: Path,
    target_layer_name: str = None
):
    """
    Generate Grad-CAM for specific labels on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor [C, H, W]
        label_indices: List of label indices to visualize
        label_columns: List of all label names
        device: Device to run on
        save_path: Path to save the visualization
        target_layer_name: Name of target layer (deprecated, auto-detected if None)
    """
    # Automatically detect target layer
    try:
        target_layer = get_target_layer(model)
    except Exception as e:
        if target_layer_name:
            # Fallback to manual layer specification
            if hasattr(model, target_layer_name):
                target_layer = getattr(model, target_layer_name)[-1]
            else:
                raise ValueError(f"Model does not have layer '{target_layer_name}'")
        else:
            raise
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Denormalize image
    img_display = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    img_pil = Image.fromarray((img_display * 255).astype(np.uint8))
    
    # Create visualization
    num_labels = len(label_indices)
    cols = min(4, num_labels)
    rows = (num_labels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for i, label_idx in enumerate(label_indices):
        row = i // cols
        col = i % cols
        
        label_name = label_columns[label_idx]
        
        # Generate CAM
        cam = gradcam.generate_cam(
            image_tensor.unsqueeze(0).to(device),
            target_label_idx=label_idx
        )
        
        # Overlay
        superimposed = overlay_heatmap_on_image(img_pil, cam, alpha=0.4)
        
        axes[row, col].imshow(superimposed)
        axes[row, col].axis('off')
        axes[row, col].set_title(label_name, fontsize=10, fontweight='bold')
    
    # Hide unused subplots
    for i in range(len(label_indices), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
