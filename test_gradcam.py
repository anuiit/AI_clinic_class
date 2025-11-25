"""
Standalone script to test Grad-CAM implementation on a trained model.
"""
import torch
from pathlib import Path
from PIL import Image
from data import build_default_transforms
from model import build_resnet18_multilabel
from gradcam import (
    GradCAM, 
    visualize_gradcam_for_specific_labels,
    overlay_heatmap_on_image,
    get_target_layer
)
import numpy as np
import matplotlib.pyplot as plt

# Configuration
CHECKPOINT_PATH = "best_resnet18_codex.pt"  # Update this to your checkpoint
IMAGE_PATH = "glyphs/Sorted/acatl-element/acatl-element_001.png"  # Update with your test image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_gradcam():
    """Test Grad-CAM on a single image."""
    
    print("=" * 60)
    print("TESTING GRAD-CAM IMPLEMENTATION")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not Path(CHECKPOINT_PATH).exists():
        print(f"\n❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH in this script to point to a valid checkpoint.")
        return
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ Image not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH in this script to point to a valid image.")
        return
    
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    label_columns = checkpoint['label_columns']
    num_labels = len(label_columns)
    
    print(f"Number of labels: {num_labels}")
    print(f"Using device: {DEVICE}")
    
    # Build model
    model = build_resnet18_multilabel(num_labels=num_labels, dropout=0.3, hidden=256).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Model loaded successfully")
    
    # Load and preprocess image
    print(f"\nLoading image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH).convert("RGB")
    transform = build_default_transforms()
    input_tensor = transform(img).to(DEVICE)
    
    print(f"Image shape: {input_tensor.shape}")
    
    # Get predictions
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()
    
    # Get top 5 predictions
    top_indices = np.argsort(probs)[::-1][:5]
    
    print("\n" + "=" * 60)
    print("TOP 5 PREDICTIONS:")
    print("=" * 60)
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {label_columns[idx]:<30} {probs[idx]:.4f}")
    
    # Initialize Grad-CAM
    print("\n" + "=" * 60)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("=" * 60)
    
    target_layer = get_target_layer(model)
    print(f"Using target layer: {target_layer.__class__.__name__}")
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAMs for top predictions
    output_dir = Path("gradcam_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Denormalize image for display
    img_display = input_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    img_pil = Image.fromarray((img_display * 255).astype(np.uint8))
    
    # Create figure with all top predictions
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    # First column: original image
    axes[0, 0].imshow(img_display)
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original\nImage', fontweight='bold', fontsize=10)
    axes[1, 0].axis('off')
    
    # Generate Grad-CAM for each top prediction
    for i, label_idx in enumerate(top_indices):
        label_name = label_columns[label_idx]
        prob = probs[label_idx]
        
        print(f"\n  Generating CAM for: {label_name} (prob: {prob:.4f})")
        
        # Generate CAM
        cam = gradcam.generate_cam(
            input_tensor.unsqueeze(0),
            target_label_idx=label_idx
        )
        
        # Overlay on image
        superimposed = overlay_heatmap_on_image(img_pil, cam, alpha=0.4)
        
        # Plot overlay
        axes[0, i + 1].imshow(superimposed)
        axes[0, i + 1].axis('off')
        axes[0, i + 1].set_title(
            f"{label_name}\nProb: {prob:.3f}",
            fontsize=9,
            fontweight='bold'
        )
        
        # Plot heatmap
        axes[1, i + 1].imshow(cam, cmap='jet')
        axes[1, i + 1].axis('off')
        axes[1, i + 1].set_title('Activation', fontsize=8)
    
    plt.suptitle('Grad-CAM Test - Top 5 Predictions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'gradcam_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Visualization saved to: {output_path}")
    
    # Also test the specific labels function
    print("\nGenerating specific labels visualization...")
    visualize_gradcam_for_specific_labels(
        model=model,
        image_tensor=input_tensor,
        label_indices=top_indices.tolist(),
        label_columns=label_columns,
        device=DEVICE,
        save_path=output_dir / 'gradcam_specific_labels.png'
        # target_layer_name auto-detected
    )
    
    print(f"✓ Specific labels visualization saved to: {output_dir / 'gradcam_specific_labels.png'}")
    
    print("\n" + "=" * 60)
    print("✅ GRAD-CAM TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput files in: {output_dir.absolute()}")
    print("  - gradcam_test.png")
    print("  - gradcam_specific_labels.png")

if __name__ == "__main__":
    try:
        test_gradcam()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
