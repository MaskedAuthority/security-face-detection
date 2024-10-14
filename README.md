# Security Face Detection

**Security Face Detection** is a face recognition system built using MediaPipe's FaceMesh technology. It captures and processes facial landmarks to create unique face signatures, which are then matched for authentication or identification purposes. The system is designed to offer a reliable method for real-time face recognition using webcam input.

## Features
- **Face Landmark Detection**: Uses MediaPipe's FaceMesh to extract 468 or 478 facial landmarks for high-accuracy recognition.
- **Face Signature Generation**: Generates a unique signature based on facial landmark coordinates.
- **Real-time Face Matching**: Compares the live face signature against a stored signature for authentication.
- **Adjustable Sensitivity**: Configurable matching threshold for varying levels of accuracy.
- **Flexibility**: Works with 468 (standard) or 478 (refined) landmarks depending on your needs.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- NumPy
- Pickle (for storing/loading face signatures)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/security-face-detection.git
   cd security-face-detection
