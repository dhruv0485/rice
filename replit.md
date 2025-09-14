# Rice Leaf Disease Detection App

## Overview

This is a Streamlit web application that uses a pre-trained deep learning model to detect diseases in rice leaves. The system analyzes uploaded images and classifies them into four categories: Bacterial Leaf Blight, Brown Spot, Leaf Smut, or Healthy/None. The application provides a user-friendly drag-and-drop interface with detailed disease information including symptoms, treatments, and severity levels.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based interface with drag-and-drop image upload functionality
- **Layout**: Wide layout configuration for better image display and results presentation
- **User Interface Components**: File uploader, image display, results panels, and disease information cards

### Backend Architecture
- **Machine Learning Pipeline**: TensorFlow/Keras-based image classification system
- **Image Processing**: PIL and OpenCV libraries for image preprocessing and validation
- **Model Architecture**: Pre-trained CNN model expecting 224x224x3 RGB input images with 4-class output
- **Caching Strategy**: Streamlit's `@st.cache_resource` decorator for efficient model loading

### Data Processing
- **Image Validation**: Automatic detection of non-rice leaf images with "Invalid Image" responses
- **Preprocessing Pipeline**: Image resizing, normalization, and format conversion for model input
- **Classification Output**: Confidence scores and disease predictions with detailed information lookup

### Application Logic
- **Disease Information System**: Comprehensive database of disease details including symptoms, treatments, and severity ratings
- **Error Handling**: Graceful handling of missing model files and invalid image uploads
- **Results Display**: Professional presentation of predictions with actionable recommendations

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for the user interface
- **TensorFlow/Keras**: Deep learning framework for model loading and inference
- **PIL (Pillow)**: Image processing and manipulation
- **OpenCV**: Computer vision operations and image validation
- **NumPy**: Numerical computations for array operations

### Model Assets
- **Pre-trained Model**: `keras_model_1757878542386.h5` - CNN model for rice disease classification
- **Label Mapping**: `labels_1757878542387.txt` - Class labels for the 4 disease categories

### Deployment Requirements
- **Python Runtime**: Version 3.7 or higher
- **Port Configuration**: Default Streamlit server port 5000
- **File System**: Local storage for model assets in `attached_assets/` directory