from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import numpy as np
import cv2
import os
import tempfile
import pathlib
import traceback as tb
import asyncio
import json
import base64
from typing import Optional

from ultralytics import YOLO
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────
# LOAD MODELS (graceful)
# ──────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── YOLO ──
yolo_model = None
yolo_error = None
try:
    yolo_model = YOLO(os.path.join(MODELS_DIR, "best.pt"))
    print("[CrowdGuard] ✓ YOLOv8 loaded")
except Exception as e:
    yolo_error = str(e)
    print(f"[CrowdGuard] ✗ YOLOv8 failed: {e}")

# ── Keras / TF models ──
def _load_keras(name: str, path: str):
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    def s1():
        from tensorflow.keras.models import load_model
        return load_model(path, compile=False)
    def s2():
        from tensorflow.keras.models import load_model
        try:
            return load_model(path, compile=False, safe_mode=False)
        except TypeError:
            raise
    def s3():
        from tensorflow.keras.models import load_model
        import tensorflow as tf
        custom = {"Orthogonal": tf.keras.initializers.Orthogonal}
        with tf.keras.utils.custom_object_scope(custom):
            return load_model(path, compile=False)
    def s4():
        import tensorflow as tf
        return tf.saved_model.load(path)
    strategies = [("standard", s1), ("safe_mode=False", s2),
                  ("custom_object_scope", s3), ("tf.saved_model", s4)]
    last_err = None
    for label, fn in strategies:
        try:
            m = fn()
            print(f"[CrowdGuard] ✓ {name} loaded via [{label}]")
            return m, None
        except Exception as e:
            last_err = f"[{label}] {e}"
    return None, last_err

lstm_model, lstm_error = _load_keras("LSTM model", os.path.join(MODELS_DIR, "lstm_density_model.h5"))

NEW_KERAS_MODEL_DIR = os.path.join(MODELS_DIR, "model_epoch_50_keras")

def _load_keras3_model(name: str, model_dir: str):
    if not os.path.isdir(model_dir):
        return None, f"Directory not found: {model_dir}"
    try:
        import keras
        model = keras.saving.load_model(model_dir, compile=False)
        print(f"[CrowdGuard] ✓ {name} loaded via keras.saving.load_model")
        return model, None
    except Exception as e1:
        pass
    try:
        from tensorflow.keras.models import load_model as lm
        model = lm(model_dir, compile=False)
        print(f"[CrowdGuard] ✓ {name} loaded via tf.keras.models.load_model")
        return model, None
    except Exception as e2:
        err = f"keras.saving: {e1} | tf.keras: {e2}"
        print(f"[CrowdGuard] ✗ {name} failed: {err}")
        return None, err

cnn_model, cnn_error = _load_keras3_model("BehaviourNet-v2", NEW_KERAS_MODEL_DIR)
if cnn_model is None:
    cnn_model, cnn_error = _load_keras("BehaviourNet-legacy", os.path.join(MODELS_DIR, "crowd_panic_cnn_lstm.h5"))

def _build_class_map(model):
    try:
        n = model.output_shape[-1]
    except Exception:
        n = 4
    if n == 2:
        return ["NORMAL", "PANIC"]
    elif n == 3:
        return ["NORMAL", "GATHERING", "PANIC"]
    else:
        return ["NORMAL", "GATHERING", "FIGHT", "PANIC"]

CNN_CLASS_LABELS = _build_class_map(cnn_model) if cnn_model is not None else ["NORMAL", "GATHERING", "FIGHT", "PANIC"]
print(f"[CrowdGuard]   CNN class labels: {CNN_CLASS_LABELS}")

scaler = None
scaler_error = None
try:
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    print("[CrowdGuard] ✓ Scaler loaded")
except Exception as e:
    scaler_error = str(e)
    print(f"[CrowdGuard] ✗ Scaler failed: {e}")

CNN_FRAMES = 20
CNN_SIZE   = (128, 128)

def _probe_cnn_shape(model):
    global CNN_FRAMES, CNN_SIZE
    try:
        import tensorflow as tf
        s = None
        if hasattr(model, "input_shape"):
            s = model.input_shape
        if s is not None:
            if len(s) == 5 and s[2] and s[3]:
                CNN_FRAMES = int(s[1]) if s[1] else CNN_FRAMES
                CNN_SIZE   = (int(s[3]), int(s[2]))
                return
            elif len(s) == 4 and s[1] and s[2]:
                CNN_SIZE = (int(s[2]), int(s[1]))
                return
        candidate_sizes = [16,24,28,32,40,48,56,60,64,72,80,84,88,96,100,104,108,112,120,128]
        for sz in candidate_sizes:
            try:
                dummy = np.zeros((1, sz, sz, 3), dtype=np.float32)
                model(dummy, training=False)
                CNN_SIZE = (sz, sz)
                return
            except Exception:
                pass
            try:
                dummy5 = np.zeros((1, CNN_FRAMES, sz, sz, 3), dtype=np.float32)
                model(dummy5, training=False)
                CNN_SIZE = (sz, sz)
                return
            except Exception:
                pass
    except Exception as e:
        print(f"[CrowdGuard]   CNN probe error: {e}")

if cnn_model is not None:
    _probe_cnn_shape(cnn_model)

TIMESTEPS  = 10
N_FEATURES = 6

def _probe_lstm_shape(model):
    global TIMESTEPS, N_FEATURES
    try:
        if hasattr(model, 'input_shape'):
            s = model.input_shape
            if len(s) == 3:
                TIMESTEPS  = int(s[1]) if s[1] else TIMESTEPS
                N_FEATURES = int(s[2]) if s[2] else N_FEATURES
    except Exception:
        pass

if lstm_model is not None:
    _probe_lstm_shape(lstm_model)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

# ──────────────────────────────────────────
# YOLO CONFIDENCE & FILTER SETTINGS
# ──────────────────────────────────────────
YOLO_CONF_THRESHOLD  = 0.35   # minimum confidence — raise to 0.45 if still noisy
YOLO_IOU_THRESHOLD   = 0.45   # NMS IOU
YOLO_PERSON_CLASS_ID = 0      # COCO class 0 = person; adjust if your custom model uses a different ID
# Max fraction of image area a person box can occupy (rejects banners/large objects)
YOLO_MAX_BBOX_AREA_FRAC = 0.25

# ──────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────

def is_video(filename: str) -> bool:
    return pathlib.Path(filename).suffix.lower() in VIDEO_EXTS


def decode_image_bytes(contents: bytes, filename: str = ""):
    np_arr = np.frombuffer(contents, np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    suffix = pathlib.Path(filename).suffix.lower() if filename else ".jpg"
    if suffix not in IMAGE_EXTS:
        suffix = ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        image = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return image


def extract_frames_from_video(contents: bytes, filename: str, n_frames: int = 20):
    suffix = pathlib.Path(filename).suffix.lower() or ".mp4"
    tmp_path = None
    frames = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        cap     = cv2.VideoCapture(tmp_path)
        total   = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    if not frames:
        return None, []
    return frames[len(frames) // 2], frames


def _is_valid_person_box(x1, y1, x2, y2, img_h, img_w):
    """
    Returns True if the bounding box looks like a person:
    - Not larger than YOLO_MAX_BBOX_AREA_FRAC of image area (rejects banners)
    - Aspect ratio: height should be taller than wide (person silhouette)
    """
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    box_area  = bw * bh
    img_area  = img_h * img_w
    area_frac = box_area / img_area
    if area_frac > YOLO_MAX_BBOX_AREA_FRAC:
        return False   # too large — likely a banner/background object
    aspect = bh / bw  # person: tall > wide, expect aspect >= 0.8
    if aspect < 0.5:
        return False   # very wide short box → not a person
    return True


def detect_people_with_boxes(image):
    """
    Run YOLOv8 and return (count, boxes_list).
    boxes_list: list of {x, y, w, h, conf, label} in pixel coords.
    Filters to person class only and rejects implausible boxes.
    """
    if yolo_model is None:
        raise RuntimeError(f"YOLOv8 not loaded: {yolo_error}")

    img_h, img_w = image.shape[:2]

    results = yolo_model(
        image,
        conf=YOLO_CONF_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        verbose=False,
        agnostic_nms=True,   # class-agnostic NMS avoids duplicate suppression issues
    )

    boxes_out = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])

            # ── PERSON-ONLY FILTER ──
            # If your custom best.pt only has 1 class, cls_id will always be 0.
            # If it has multiple classes, only keep class 0 (person).
            if cls_id != YOLO_PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if not _is_valid_person_box(x1, y1, x2, y2, img_h, img_w):
                continue

            boxes_out.append({
                "x":    round(x1),
                "y":    round(y1),
                "w":    round(x2 - x1),
                "h":    round(y2 - y1),
                "conf": round(conf, 3),
                "label": f"P-{len(boxes_out)+1:02d}",
            })

    return len(boxes_out), boxes_out


def count_people(image) -> int:
    count, _ = detect_people_with_boxes(image)
    return count


def calculate_density(count: int, image) -> float:
    h, w = image.shape[:2]
    area = (h * w) / 10000
    return round(count / area if area > 0 else 0.0, 2)


def predict_behavior_from_frames(frames):
    if cnn_model is None:
        return "NORMAL", False
    try:
        resized = [cv2.resize(f, CNN_SIZE).astype(np.float32) / 255.0 for f in frames]
        while len(resized) < CNN_FRAMES:
            resized.append(resized[-1])
        resized = resized[:CNN_FRAMES]
        in_ndims = len(cnn_model.input_shape) if hasattr(cnn_model, "input_shape") else 5
        if in_ndims == 4:
            mid  = resized[len(resized) // 2]
            clip = np.expand_dims(mid, axis=0)
        else:
            clip = np.stack(resized, axis=0)
            clip = np.expand_dims(clip, axis=0)
        pred     = cnn_model.predict(clip, verbose=0)
        label    = int(np.argmax(pred))
        behavior = CNN_CLASS_LABELS[label] if label < len(CNN_CLASS_LABELS) else "UNKNOWN"
        return behavior, behavior in ["FIGHT", "PANIC"]
    except Exception as e:
        print(f"[CrowdGuard] BehaviourNet inference error: {e}")
        return "NORMAL", False


def forecast_lstm(count: int, density: float) -> int:
    if lstm_model is None or scaler is None:
        return max(0, round(count * 1.05))
    try:
        n_feat       = int(getattr(scaler, 'n_features_in_', N_FEATURES))
        delta        = 0.0
        in_flow      = max(count * 0.1, 0.0)
        out_flow     = max(count * 0.1, 0.0)
        delta2       = 0.0
        density_rate = density / 10000.0 if density else 0.0
        base_row     = np.array([[count, delta, in_flow, out_flow, delta2, density_rate]], dtype=np.float32)
        if base_row.shape[1] < n_feat:
            pad      = np.zeros((1, n_feat - base_row.shape[1]), dtype=np.float32)
            base_row = np.concatenate([base_row, pad], axis=1)
        elif base_row.shape[1] > n_feat:
            base_row = base_row[:, :n_feat]
        row_scaled    = scaler.transform(base_row)
        seq           = np.tile(row_scaled, (TIMESTEPS, 1))
        seq           = np.expand_dims(seq, axis=0)
        pred_scaled   = lstm_model.predict(seq, verbose=0)
        inv_row       = np.zeros((1, n_feat), dtype=np.float32)
        inv_row[0, 0] = float(pred_scaled.flat[0])
        if n_feat > 1:
            inv_row[0, 1:] = row_scaled[0, 1:n_feat]
        forecast_row = scaler.inverse_transform(inv_row)
        return int(max(0, round(float(forecast_row[0, 0]))))
    except Exception as e:
        print(f"[CrowdGuard] LSTM inference error: {e}")
        return max(0, round(count * 1.05))


# ──────────────────────────────────────────
# MOBILE CAMERA RELAY  (WebSocket + HTTP fallback)
# ──────────────────────────────────────────
# Holds the current frame sent by the mobile phone (base64 JPEG)
mobile_frame_store: dict = {"frame": None}
# Desktop listeners waiting for mobile frames
desktop_listeners: list = []

# ── HTTP POST fallback: mobile uploads one frame at a time ──────────────────
@app.post("/mobile/frame")
async def mobile_frame_post(request: Request):
    """
    HTTP POST fallback for mobile cameras that cannot maintain a WebSocket.
    The phone sends a raw JPEG as the request body (application/octet-stream)
    OR as a base64 JSON field {"frame": "<b64>"}.
    This endpoint stores the frame and fans it out to desktop WS listeners.
    """
    content_type = request.headers.get("content-type", "")
    try:
        if "application/json" in content_type:
            body = await request.json()
            data = body.get("frame", "")
        else:
            # Raw bytes — encode to base64 ourselves
            raw = await request.body()
            if not raw:
                return {"ok": False, "error": "empty body"}
            data = base64.b64encode(raw).decode()

        if not data or len(data) < 100:
            return {"ok": False, "error": "frame too small"}

        mobile_frame_store["frame"] = data

        # Fan out to all connected desktop WS clients
        dead = []
        for ws in list(desktop_listeners):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            try:
                desktop_listeners.remove(ws)
            except ValueError:
                pass

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── HTTP GET: desktop polls for the latest mobile frame ─────────────────────
@app.get("/mobile/frame")
async def mobile_frame_get():
    """
    HTTP GET fallback for desktops that want to poll for the latest mobile frame
    instead of using the WebSocket relay.
    Returns {"frame": "<base64 JPEG>"} or {"frame": null}.
    """
    return {"frame": mobile_frame_store.get("frame")}

@app.websocket("/ws/mobile_send")
async def ws_mobile_send(websocket: WebSocket):
    """Mobile phone connects here to stream camera frames."""
    await websocket.accept()
    print("[CrowdGuard] 📱 Mobile camera connected")
    try:
        while True:
            # Accept both text and binary messages so the handler never
            # crashes on an unexpected frame type.
            try:
                msg = await websocket.receive()
            except Exception as recv_err:
                print(f"[CrowdGuard] 📱 Mobile receive error: {recv_err}")
                break

            # Normalise to a string regardless of how the frame arrived.
            if msg["type"] == "websocket.disconnect":
                break
            elif msg["type"] == "websocket.receive":
                data = msg.get("text") or ""
                if not data and msg.get("bytes"):
                    # Binary frame — shouldn't happen but handle gracefully
                    import base64 as _b64
                    data = _b64.b64encode(msg["bytes"]).decode()
            else:
                continue

            # Ignore keep-alive pings / very short messages
            if not data or data == "ping" or len(data) < 100:
                continue

            # Store latest frame and relay to all desktop listeners.
            mobile_frame_store["frame"] = data

            # Iterate over a snapshot to avoid mutation-during-iteration bugs.
            dead = []
            for ws in list(desktop_listeners):
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                try:
                    desktop_listeners.remove(ws)
                except ValueError:
                    pass  # already removed by another coroutine

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CrowdGuard] 📱 Mobile camera error: {e}")
    finally:
        print("[CrowdGuard] 📱 Mobile camera disconnected")
        mobile_frame_store["frame"] = None


@app.websocket("/ws/mobile_recv")
async def ws_mobile_recv(websocket: WebSocket):
    """Desktop dashboard connects here to receive mobile frames."""
    await websocket.accept()
    desktop_listeners.append(websocket)
    print("[CrowdGuard] 🖥  Desktop subscribed to mobile feed")

    # If a frame is already buffered, push it immediately so the desktop
    # doesn't wait for the next phone frame cycle.
    if mobile_frame_store["frame"]:
        try:
            await websocket.send_text(mobile_frame_store["frame"])
        except Exception:
            pass

    try:
        while True:
            # Use the low-level receive() so we handle both text (pings) and
            # binary frames without crashing, and also detect clean disconnects.
            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=30)
            except asyncio.TimeoutError:
                # No ping received within 30 s — send a keep-alive ping back.
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break  # Connection is dead; exit loop to clean up.
                continue

            if msg["type"] == "websocket.disconnect":
                break
            # Silently ignore any text/binary data sent by the desktop client.
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CrowdGuard] 🖥  Desktop recv error: {e}")
    finally:
        try:
            desktop_listeners.remove(websocket)
        except ValueError:
            pass
        print("[CrowdGuard] 🖥  Desktop unsubscribed from mobile feed")


@app.get("/mobile", response_class=HTMLResponse)
async def mobile_page():
    """Serve the mobile camera page."""
    html_path = os.path.join(BASE_DIR, "frontend", "mobile.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>mobile.html not found</h1>", status_code=404)


# ──────────────────────────────────────────
# API ENDPOINTS
# ──────────────────────────────────────────

@app.get("/server-info")
async def server_info(request: Request):
    """Return the server's LAN IP so the frontend can build the correct mobile URL."""
    import socket

    def _udp_trick():
        """Classic UDP trick — connects a UDP socket to determine the outbound interface IP."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and not ip.startswith("127."):
                return ip
        except Exception:
            pass
        return None

    def _netifaces_scan():
        """Walk every network interface and return the first routable LAN IP."""
        try:
            import netifaces
            # Prefer wlan/wifi/eth interfaces; skip loopback
            preferred = []
            fallback  = []
            for iface in netifaces.interfaces():
                if iface.startswith("lo"):
                    continue
                addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET, [])
                for addr in addrs:
                    ip = addr.get("addr", "")
                    if (ip.startswith("192.168.") or ip.startswith("10.") or
                            any(ip.startswith(f"172.{i}.") for i in range(16, 32))):
                        # Wi-Fi / Ethernet names get priority
                        if any(k in iface.lower() for k in ("wlan","wifi","wlp","eth","en","eno","enp")):
                            preferred.append(ip)
                        else:
                            fallback.append(ip)
            return (preferred or fallback or [None])[0]
        except Exception:
            pass
        return None

    def _socket_scan():
        """Enumerate all IPs bound to this hostname."""
        try:
            hostname = socket.gethostname()
            results  = socket.getaddrinfo(hostname, None, socket.AF_INET)
            for r in results:
                ip = r[4][0]
                if not ip.startswith("127."):
                    return ip
        except Exception:
            pass
        return None

    def _ifconfig_scan():
        """Parse `ip addr` / `ifconfig` output as last resort (Linux/Mac)."""
        import re, subprocess
        for cmd in (["ip", "-4", "addr", "show"], ["ifconfig"]):
            try:
                out = subprocess.check_output(cmd, timeout=3, stderr=subprocess.DEVNULL).decode()
                for ip in re.findall(r"inet\s+(\d+\.\d+\.\d+\.\d+)", out):
                    if not ip.startswith("127."):
                        return ip
            except Exception:
                pass
        return None

    # Try in order of reliability
    lan_ip = (_udp_trick() or _netifaces_scan() or _socket_scan() or _ifconfig_scan() or "127.0.0.1")

    # Grab port: prefer the Host header sent by the browser (most accurate)
    host_header = request.headers.get("host", "")
    if ":" in host_header:
        port = host_header.split(":")[-1]
    else:
        port = "8000"

    mobile_url = f"http://{lan_ip}:{port}/mobile"
    print(f"[CrowdGuard] server-info → LAN IP: {lan_ip}  port: {port}  mobile: {mobile_url}")
    return {
        "lan_ip":     lan_ip,
        "port":       port,
        "mobile_url": mobile_url,
        "is_loopback": lan_ip.startswith("127."),
    }


@app.get("/status")
async def status():
    return {
        "yolo":   {"loaded": yolo_model is not None, "error": yolo_error},
        "lstm":   {"loaded": lstm_model is not None, "error": lstm_error},
        "cnn":    {"loaded": cnn_model  is not None, "error": cnn_error},
        "scaler": {"loaded": scaler     is not None, "error": scaler_error},
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        if yolo_model is None:
            return {"error": f"YOLOv8 model not loaded — {yolo_error}."}

        contents = await file.read()
        filename = file.filename or ""

        if not contents:
            return {"error": "Received an empty file."}

        warnings = {}

        if is_video(filename):
            middle_frame, frames = extract_frames_from_video(contents, filename, n_frames=CNN_FRAMES)
            if middle_frame is None or not frames:
                return {"error": "Could not read any frames from the video."}
            count, boxes      = detect_people_with_boxes(middle_frame)
            density           = calculate_density(count, middle_frame)
            behavior, anomaly = predict_behavior_from_frames(frames)
            forecast          = forecast_lstm(count, density)
            img_h, img_w      = middle_frame.shape[:2]
        else:
            image = decode_image_bytes(contents, filename)
            if image is None:
                return {"error": f"Could not decode image '{filename}'."}
            count, boxes      = detect_people_with_boxes(image)
            density           = calculate_density(count, image)
            behavior, anomaly = predict_behavior_from_frames([image])
            forecast          = forecast_lstm(count, density)
            img_h, img_w      = image.shape[:2]

        if cnn_model  is None: warnings["behaviour"] = f"BehaviourNet unavailable; defaulted to NORMAL"
        if lstm_model is None: warnings["forecast"]  = f"LSTM unavailable; used rule-based estimate"
        if scaler     is None: warnings["forecast"]  = "Scaler unavailable; used rule-based estimate"

        return {
            "count":    count,
            "density":  density,
            "behavior": behavior,
            "anomaly":  anomaly,
            "forecast": forecast,
            "boxes":    boxes,           # ← real bounding boxes from YOLO
            "img_w":    img_w,
            "img_h":    img_h,
            "warnings": warnings,
        }

    except Exception as e:
        return {"error": str(e), "detail": tb.format_exc()}


@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Analyze a single JPEG frame from the browser or mobile webcam."""
    try:
        if yolo_model is None:
            return {"error": f"YOLOv8 not loaded: {yolo_error}"}

        contents = await file.read()
        if not contents:
            return {"error": "Empty frame received."}

        image = decode_image_bytes(contents, "frame.jpg")
        if image is None:
            return {"error": "Could not decode frame."}

        count, boxes = detect_people_with_boxes(image)
        density      = calculate_density(count, image)
        risk         = "HIGH" if count > 100 else "MED" if count > 50 else "LOW"
        img_h, img_w = image.shape[:2]

        behavior, anomaly = predict_behavior_from_frames([image])

        if cnn_model is None:
            if count > 120:
                behavior, anomaly = "PANIC", True
            else:
                behavior, anomaly = "NORMAL", False

        forecast = forecast_lstm(count, density)

        return {
            "count":    count,
            "density":  density,
            "behavior": behavior,
            "anomaly":  anomaly,
            "forecast": forecast,
            "risk":     risk,
            "boxes":    boxes,      # ← real bounding boxes
            "img_w":    img_w,
            "img_h":    img_h,
        }

    except Exception as e:
        return {"error": str(e), "detail": tb.format_exc()}


# Serve frontend
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

def _print_qr_console(url: str):
    """Print a small QR code directly in the terminal using only stdlib."""
    try:
        import urllib.request, json as _json
        # Use the free goqr.me API to get a text/unicode QR — no pip install needed.
        # Falls back gracefully if offline.
        api = f"https://api.qrserver.com/v1/create-qr-code/?data={urllib.request.quote(url)}&size=200x200&format=svg"
        print(f"\n[CrowdGuard] 📱 Mobile URL : {url}")
        print(f"[CrowdGuard] 📷 Open this URL on your phone (same Wi-Fi) or scan QR in the dashboard.\n")
    except Exception:
        pass


def _try_start_ngrok(port: int):
    """
    Attempt to start an ngrok HTTPS tunnel.
    Returns the public https:// URL string, or None if ngrok isn't installed / fails.

    Install ngrok once:  pip install pyngrok
    Or download from:    https://ngrok.com/download
    """
    # ── Strategy 1: pyngrok (pip install pyngrok) ──────────────────────────
    try:
        from pyngrok import ngrok as _ngrok
        tunnel = _ngrok.connect(port, "http")
        url = tunnel.public_url
        if url.startswith("http://"):
            url = "https://" + url[7:]   # ngrok always supports https
        print(f"[CrowdGuard] 🌐 ngrok tunnel active → {url}")
        print(f"[CrowdGuard] 📱 Mobile camera URL  → {url}/mobile")
        print(f"[CrowdGuard] ↑  Share this URL — it works from ANY network, no same-Wi-Fi needed!\n")
        return url
    except ImportError:
        pass
    except Exception as e:
        print(f"[CrowdGuard] ⚠ pyngrok error: {e}")

    # ── Strategy 2: ngrok binary on PATH ──────────────────────────────────
    import subprocess, threading, time, re
    try:
        proc = subprocess.Popen(
            ["ngrok", "http", str(port), "--log=stdout", "--log-format=json"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        # Read lines until we see the tunnel URL (give it 8 seconds)
        url_found = []
        def _reader():
            for line in proc.stdout:
                m = re.search(r'"url":"(https://[^"]+)"', line)
                if m:
                    url_found.append(m.group(1))
                    break
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=8)
        if url_found:
            url = url_found[0]
            print(f"[CrowdGuard] 🌐 ngrok tunnel active → {url}")
            print(f"[CrowdGuard] 📱 Mobile camera URL  → {url}/mobile")
            print(f"[CrowdGuard] ↑  Share this URL — it works from ANY network!\n")
            return url
    except FileNotFoundError:
        pass   # ngrok not installed
    except Exception as e:
        print(f"[CrowdGuard] ⚠ ngrok binary error: {e}")

    return None


# ── Expose the ngrok URL via /server-info so the dashboard QR updates ────────
_ngrok_url: str | None = None

_original_server_info = None
for _r in app.routes:
    if hasattr(_r, "path") and _r.path == "/server-info":
        _original_server_info = _r.endpoint
        break

if _original_server_info:
    app.routes[:] = [r for r in app.routes if not (hasattr(r, "path") and r.path == "/server-info")]

    @app.get("/server-info")
    async def server_info_with_ngrok(request: Request):
        base = await _original_server_info(request)
        if _ngrok_url:
            base["ngrok_url"]   = _ngrok_url
            base["mobile_url"]  = _ngrok_url + "/mobile"
            base["is_loopback"] = False
        return base


if __name__ == "__main__":
    PORT = 8000

    # Try to start ngrok tunnel for hassle-free mobile access
    print("[CrowdGuard] 🔌 Attempting to start ngrok tunnel for mobile access…")
    print("[CrowdGuard]    (If ngrok isn't installed: pip install pyngrok)")
    _ngrok_url = _try_start_ngrok(PORT)

    if not _ngrok_url:
        print("[CrowdGuard] ℹ  ngrok not available — using LAN IP only.")
        print("[CrowdGuard]    Phone must be on the SAME Wi-Fi as this PC.")
        print("[CrowdGuard]    To get a universal URL: pip install pyngrok\n")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
