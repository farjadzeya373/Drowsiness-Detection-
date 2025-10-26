# Drowsiness Detection (Windows) - Full Project
This project provides a Windows-ready drowsiness detection system that uses a CNN trained on the **MRL Eye Dataset** (open/closed eyes). It includes:

- `download_datasets.py` — auto-download (and extract) helper for MRL dataset (you may need to edit the download URL if required).
- `train_eye_model.py` — trains a lightweight CNN on the eye images and saves `model/eye_model.h5`.
- `drowsiness_detection.py` — real-time webcam detector using the trained model and `winsound` for alerts.
- `requirements.txt` — Python dependencies.
- `dataset/` — placeholder for downloaded data.
- `model/` — where the trained model will be saved.

## Quick start (Windows)
1. Create and activate a Python environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Download the dataset automatically (recommended):
   ```bash
   python download_datasets.py
   ```
   - The script attempts to download only the necessary eye images from the MRL Eye Dataset and extract them into `dataset/eyes/open` and `dataset/eyes/closed`.
   - If automatic download fails (site may require manual access), follow manual instructions inside the script or the MRL dataset page and place images into `dataset/eyes/open` and `dataset/eyes/closed`.
3. Train the eye classifier (takes minutes to hours depending on your GPU/CPU):
   ```bash
   python train_eye_model.py
   ```
   This saves `model/eye_model.h5`.
4. Run real-time detection:
   ```bash
   python drowsiness_detection.py
   ```
   Press `ESC` to quit. Logs will be appended to `drowsiness_log.csv`.

## Notes
- This Windows build uses `winsound.Beep` for alert beeps.
- For improved accuracy consider using MediaPipe face mesh; see comments in `drowsiness_detection.py`.
- The download script includes a fallback/manual option if the MRL site prevents automated downloading.
