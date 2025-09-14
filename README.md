# Rice Leaf Disease Detection App

A Streamlit web application for detecting diseases in rice leaves using a pre-trained deep learning model.

## Features

- **Image Upload**: Drag-and-drop interface for rice leaf images
- **Disease Detection**: AI-powered classification of 4 disease categories:
  - Bacterial Leaf Blight
  - Brown Spot
  - Leaf Smut
  - Healthy/None
- **Image Validation**: Detects non-rice leaf images and shows 'Invalid Image' message
- **Detailed Results**: Shows predicted disease with confidence score
- **Disease Information**: Comprehensive details about symptoms, treatments, and severity
- **Professional Interface**: Clean, user-friendly design with recommendations

## Requirements

The application requires the following files to be present:

1. **Model File**: `attached_assets/keras_model_1757878542386.h5`
   - Pre-trained Keras model for rice disease classification
   - Expected input: 224x224x3 RGB images
   - Expected output: 4 classes

2. **Labels File**: `attached_assets/labels_1757878542387.txt`
   - Contains class labels mapping (already included)

## Installation & Setup

1. Ensure you have Python 3.7+ installed
2. Install required dependencies:
   ```bash
   pip install streamlit tensorflow opencv-python pillow numpy
   ```

3. **Important**: Place your trained Keras model file (`keras_model_1757878542386.h5`) in the `attached_assets/` directory

4. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   