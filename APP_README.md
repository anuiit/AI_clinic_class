# Model Inference Web App

A minimal web application for running inference on trained models with Grad-CAM visualizations.

## Features

- **Image Upload**: Upload images via click or drag-and-drop
- **Auto Resize**: Images are automatically resized to model input size (224x224)
- **Inference**: Real-time prediction with adjustable confidence threshold
- **Predictions**: Shows all labels above the selected threshold percentage
- **Grad-CAM Visualization**: Visual explanation highlighting model focus areas
- **History**: Stores last 50 inference results with timestamps for review

## Installation

Install Flask if not already installed:
```bash
pip install flask
```

All other dependencies (torch, torchvision, cv2, etc.) should already be installed in your environment.

## Usage

### Windows
Use the provided batch script:
```bash
start_app.bat runs/run_50
```

### Manual Start
Activate your conda environment and run:
```bash
conda activate torchgpu
python app.py runs/run_50
```

Replace `runs/run_50` with any other run folder you want to use.

Then open your browser to: **http://localhost:5000**

## How to Use the App

1. **Upload an Image**: 
   - Click the upload area or drag and drop an image (PNG, JPG up to 16MB)
   
2. **Set Threshold**: 
   - Adjust the slider to set minimum confidence percentage (0-100%)
   - Default is 50%
   - Only predictions above this threshold will be shown

3. **Analyze**: 
   - Click the "Analyze" button to run inference
   
4. **View Results**: 
   - **Left side**: Grad-CAM heatmap overlay showing where the model is focusing
   - **Right side**: List of predicted labels with confidence bars
   
5. **History**: 
   - Click any history item to view its results again
   - Last 50 analyses are saved

## Technical Details

- Model and configuration loaded from the specified run folder
- Images automatically resized to model input size
- Grad-CAM visualization generated for the top prediction
- History saved to `history.json` (auto-created)
- Uploaded images stored in `uploads/` folder (auto-created)
- Compatible with models trained using different architectures (ResNet, EfficientNet, VGG, ViT)

## Notes

- The app handles checkpoint format variations automatically
- Works with models trained with different head configurations (with/without normalization layers)
- Supports models with custom bottleneck dimensions
- GPU will be used if available, otherwise falls back to CPU
