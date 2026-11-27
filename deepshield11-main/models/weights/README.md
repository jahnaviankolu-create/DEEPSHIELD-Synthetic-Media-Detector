# DeepShield Trained AI Models Weight Directory

Place your trained PyTorch (`.pt` / `.pth`) or ONNX (`.onnx`) model files in this directory.

## Expected Naming Conventions:

1. **Image Deepfake Detection Model**:
   - `image_model.pt`
   - `image_model.pth`
   - `image_model.onnx`

2. **Video Deepfake Detection Model**:
   - `video_model.pt`
   - `video_model.pth`
   - `video_model.onnx`

3. **Audio Deepfake Detection Model**:
   - `audio_model.pt`
   - `audio_model.pth`
   - `audio_model.onnx`

## Automated Weight Loading:
When a trained model file is detected in this directory, DeepShield's `models.loader` automatically loads the model weights for inference. If no weights file is present, the API safely returns `"Model not yet trained."`.
