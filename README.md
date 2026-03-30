# Eye Strain & Blink Rate Monitor with Smart Fatigue Detection

Real-time webcam-based Python project that detects eye blinks and fatigue signals using MediaPipe Face Mesh and OpenCV.

## Features

- Real-time webcam feed with live dashboard overlay
- Face and eye landmark detection (MediaPipe Face Mesh)
- EAR-based blink detection
- Blink count and blink rate (per minute)
- Eye closure duration tracking
- Fatigue Score (0-100)
- Eye Health Score (0-100)
- Distance-from-screen estimate from face area
- Session timer
- Alerts:
  - Blink more!
  - Take a break!
  - Eyes closed too long!
  - Move closer/farther
- CSV logging of session metrics
- Lightweight defaults for low-end laptops

## Project Structure

- `eye_strain_monitor.py` : Main runnable application
- `requirements.txt` : Python dependencies
- `.gitignore` : Excludes heavy and unnecessary files from git
- `logs/` : Runtime session CSV output (auto-created, ignored by git)

## Prerequisites

- Windows/Linux/macOS
- Python 3.10 or 3.11 recommended
- Webcam

## Setup (Command Line Only)

Run all commands from the project root.

### 1) Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

### 2) Activate virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```powershell
pip install -r requirements.txt
```

## Run the Project

```powershell
python eye_strain_monitor.py
```

- Press `q` to quit.
- A session CSV log is generated in `logs/`.

## Config (Optional)

Edit tunable defaults in `Config` inside `eye_strain_monitor.py`:

- `camera_index` : Webcam index (0/1/2)
- `frame_width`, `frame_height` : Resolution
- `process_every_n_frames` : Inference frequency (higher = lighter)
- `break_reminder_minutes` : Break interval
- `target_blink_rate_min`, `target_blink_rate_max` : Blink target range

## Output Metrics

Dashboard shows:

- Blink Count
- Blink Rate (/min)
- Fatigue Score
- Eye Health Score
- Distance Label
- Timer
- Alert Message

CSV fields include:

- timestamp
- session_seconds
- ear
- blink_count
- blink_rate_per_min
- closure_duration_sec
- fatigue_score
- eye_health_score
- distance_label
- face_area_ratio
- alert

## Troubleshooting

### Webcam not opening

- Close apps using camera (Zoom/Meet/Camera app)
- Change `camera_index` in `Config` from `0` to `1` or `2`
- Run again

### Module import errors

- Ensure venv is activated
- Reinstall dependencies:

```powershell
pip install -r requirements.txt
```

### Slow performance

- Lower resolution in `Config` (for example 512x288)
- Increase `process_every_n_frames` (for example 3)
- Close heavy background applications
