# CrowdGuard AI — Crowd Surveillance System

## Fixes in this version

### 1. YOLOv8 Bounding Boxes — Only People Detected
- **Problem:** YOLO was drawing boxes on banners, signs, and other objects.
- **Fix:** Added strict filtering in `main.py`:
  - Only class 0 (person) is kept. All other classes are discarded.
  - Confidence threshold: `0.35` (raise to `0.45` in `main.py` if still noisy).
  - Area filter: boxes larger than 25% of the image (banners, backgrounds) are rejected.
  - Aspect ratio filter: boxes wider than tall are rejected (persons are taller than wide).
  - Real bounding box coordinates are now returned by the API and drawn accurately on screen.

### 2. Mobile Phone Camera Support
- **How it works:**
  1. On the desktop dashboard → **Live tab** → change **Source** to **"📱 Mobile Phone Camera"**.
  2. A QR code appears. Scan it with your phone (or open the URL shown).
  3. On your phone, tap **Start Streaming** — the phone's camera streams to the server via WebSocket.
  4. Click **START** on the desktop. The dashboard analyses frames from your phone camera.
- The mobile page (`/mobile`) works in any modern mobile browser — no app required.
- Supports front and back camera toggle on the phone.

## Running

```bash
pip install -r requirements.txt
python main.py
```

Then open: http://localhost:8000

For mobile camera access on the same network:
- Your phone must be on the same Wi-Fi as the server.
- Open: http://<server-ip>:8000/mobile

## Tuning YOLO false positives
Edit `main.py` and adjust:
```python
YOLO_CONF_THRESHOLD  = 0.35   # raise to 0.5 to reduce false positives
YOLO_MAX_BBOX_AREA_FRAC = 0.25  # lower to 0.15 to reject more large objects
```
