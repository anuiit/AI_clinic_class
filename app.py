import os
import json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from pathlib import Path
from datetime import datetime
import io
import base64

from torchvision import transforms
from model import MultiLabelModel
from gradcam import GradCAM, get_target_layer
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['HISTORY_FILE'] = 'history.json'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
label_columns = None
config = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model_from_run(run_folder):
    """Load model and config from a run folder"""
    global model, label_columns, config
    
    run_path = Path(run_folder)
    config_path = run_path / 'config.json'
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Find model file
    model_files = list(run_path.glob('*.pt'))
    if not model_files:
        raise FileNotFoundError(f"No model file found in {run_folder}")
    model_path = model_files[0]
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle both checkpoint dict and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Get label columns from checkpoint if available
        if 'label_columns' in checkpoint:
            label_columns = checkpoint['label_columns']
            print(f"Loaded {len(label_columns)} labels from checkpoint")
        else:
            # Fallback: load from CSV
            import pandas as pd
            csv_path = config['csv_path']
            df = pd.read_csv(csv_path)
            non_label_cols = ['full_image_path', 'glyph_cote', 'elements_original', 
                              'codex', 'glyph_image', 'Unnamed: 0']
            non_label_cols = [c for c in non_label_cols if c in df.columns]
            label_columns = [c for c in df.columns if c not in non_label_cols]
            print(f"Loaded {len(label_columns)} labels from CSV (fallback)")
    else:
        # Old format: direct state_dict, load labels from CSV
        state_dict = checkpoint
        import pandas as pd
        csv_path = config['csv_path']
        df = pd.read_csv(csv_path)
        non_label_cols = ['full_image_path', 'glyph_cote', 'elements_original', 
                          'codex', 'glyph_image', 'Unnamed: 0']
        non_label_cols = [c for c in non_label_cols if c in df.columns]
        label_columns = [c for c in df.columns if c not in non_label_cols]
        print(f"Loaded {len(label_columns)} labels from CSV (old format)")
    
    # Initialize model
    hidden = config.get('hidden', [])
    if isinstance(hidden, int):
        hidden = [hidden]
    
    # Handle activation
    activation_str = config.get('activation', 'ReLU(inplace=True)')
    if 'ReLU' in activation_str:
        activation = nn.ReLU(inplace=True)
    elif 'LeakyReLU' in activation_str:
        activation = nn.LeakyReLU(inplace=True)
    else:
        activation = nn.ReLU(inplace=True)
    
    # Handle normalization
    normalization_str = config.get('normalization', None)
    normalization = None
    if normalization_str and 'BatchNorm1d' in str(normalization_str):
        normalization = nn.BatchNorm1d
    elif normalization_str and 'LayerNorm' in str(normalization_str):
        normalization = nn.LayerNorm
    
    model = MultiLabelModel(
        base_model=config['model'],
        num_labels=len(label_columns),
        dropout=config['dropout'],
        hidden=hidden,
        batch_size=config['batch_size'],
        activation=activation,
        normalization=normalization,
        custom_head=config.get('custom_head', False),
        use_bottleneck=config.get('use_bottleneck', False),
        bottleneck_dim=config.get('bottleneck_dim', 1024)
    )
    
    # Fix for checkpoint compatibility: older checkpoints may have different head structure
    # Check if checkpoint final layer matches current model
    checkpoint_head_keys = [k for k in state_dict.keys() if k.startswith('head.')]
    if checkpoint_head_keys:
        max_head_idx = max([int(k.split('.')[1]) for k in checkpoint_head_keys if k.split('.')[1].isdigit()])
        final_layer_key = f'head.{max_head_idx}.weight'
        
        # If checkpoint's final layer is Linear(num_labels, hidden_dim) but model expects more layers
        if final_layer_key in state_dict:
            checkpoint_final_shape = state_dict[final_layer_key].shape
            if checkpoint_final_shape[0] == len(label_columns):
                # Checkpoint has final layer at a different index than current model
                # Find the final layer in current model
                current_state = model.state_dict()
                current_head_keys = [k for k in current_state.keys() if k.startswith('head.')]
                current_max_idx = max([int(k.split('.')[1]) for k in current_head_keys if k.split('.')[1].isdigit()])
                
                if max_head_idx != current_max_idx:
                    # Remap the final layer
                    print(f"Remapping head.{max_head_idx} -> head.{current_max_idx}")
                    state_dict[f'head.{current_max_idx}.weight'] = state_dict.pop(f'head.{max_head_idx}.weight')
                    state_dict[f'head.{current_max_idx}.bias'] = state_dict.pop(f'head.{max_head_idx}.bias')
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {run_folder}")
    print(f"Labels: {len(label_columns)}")


def get_transform(img_size=224):
    """Get image transform pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict(image_path, threshold=0.5):
    """Run inference on an image"""
    img_size = 224  # Default, can be adjusted based on model
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(img_size)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Filter by threshold
    predictions = []
    for i, (label, prob) in enumerate(zip(label_columns, probs)):
        if prob >= threshold:
            predictions.append({
                'label': label,
                'confidence': float(prob * 100)
            })
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return predictions, probs


def generate_gradcam(image_path, predictions):
    """Generate Grad-CAM visualizations for all predictions"""
    img_size = 224
    
    if not predictions:
        return []
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(img_size)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get target layer
    target_layer = get_target_layer(model, config['model'])
    
    # Initialize GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Load original image for overlay
    original_img = cv2.imread(str(image_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (img_size, img_size))
    
    gradcam_results = []
    
    # Generate CAM for each prediction
    for pred in predictions:
        label_idx = label_columns.index(pred['label'])
        cam = gradcam.generate_cam(img_tensor, label_idx)
        
        # Ensure CAM is the right size
        if cam.shape != (img_size, img_size):
            cam = cv2.resize(cam, (img_size, img_size))
        
        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = (cam * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure dimensions match before overlay
        if heatmap.shape != original_img.shape:
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        # Overlay
        overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        img_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        gradcam_results.append({
            'label': pred['label'],
            'confidence': pred['confidence'],
            'gradcam': img_str
        })
    
    return gradcam_results


def save_to_history(image_name, predictions, gradcam_img):
    """Save inference result to history"""
    history = load_history()
    
    entry = {
        'timestamp': datetime.now().isoformat(),
        'image': image_name,
        'predictions': predictions,
        'gradcam': gradcam_img
    }
    
    history.insert(0, entry)  # Add to beginning
    
    # Keep only last 50 entries
    history = history[:50]
    
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f, indent=2)


def load_history():
    """Load inference history"""
    if os.path.exists(app.config['HISTORY_FILE']):
        with open(app.config['HISTORY_FILE'], 'r') as f:
            return json.load(f)
    return []


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload and inference"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get threshold
    threshold = float(request.form.get('threshold', 0.5))
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Run inference
        predictions, probs = predict(filepath, threshold)
        
        # Generate Grad-CAM
        gradcam_img = generate_gradcam(filepath, predictions)
        
        # Save to history
        save_to_history(filename, predictions, gradcam_img)
        
        return jsonify({
            'predictions': predictions,
            'gradcam': gradcam_img,
            'image': filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    """Get inference history"""
    return jsonify(load_history())


@app.route('/image/<filename>')
def get_image(filename):
    """Serve uploaded image"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python app.py <run_folder>")
        print("Example: python app.py runs/run_1")
        sys.exit(1)
    
    run_folder = sys.argv[1]
    
    # Load model
    load_model_from_run(run_folder)
    
    # Run app
    app.run(debug=True, port=5000)
