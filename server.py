"""
LineCall Python backend
=======================
Receives JPEG frames over WebSocket, detects ball, calls IN/OUT.

Install:
    pip install fastapi uvicorn[standard] opencv-python-headless ultralytics numpy filterpy

Run:
    python server.py
"""

import asyncio, base64, json, logging
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from filterpy.kalman import KalmanFilter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("linecall")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ball detector ─────────────────────────────────────────
class BallDetector:
    """
    Uses YOLOv8 nano for real video.
    Falls back to a physics simulator when no real ball is found
    (useful for testing without a real court).
    """
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            log.info("YOLOv8 loaded")
        except Exception as e:
            log.warning(f"YOLOv8 not available ({e}) — using mock detector")

        # Mock ball state for testing
        self._x   = 300.0
        self._y   = 250.0
        self._vx  = 4.5
        self._vy  = -6.0

    def detect(self, frame: np.ndarray):
        """Returns (cx, cy) or None."""
        if self.model is not None:
            results = self.model(frame, classes=[32], conf=0.35, verbose=False)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best   = boxes[boxes.conf.argmax()]
                x1,y1,x2,y2 = best.xyxy[0].tolist()
                return ((x1+x2)/2, (y1+y2)/2)

        # Physics mock — simulates a bouncing ball
        h, w = frame.shape[:2]
        self._vy += 0.3
        self._x  += self._vx
        self._y  += self._vy
        if self._x <= 20 or self._x >= w-20: self._vx *= -1
        if self._y >= h-20:
            self._y  = h-20
            self._vy = -abs(self._vy) * 0.85
        if self._y <= 20:
            self._y  = 20
            self._vy = abs(self._vy) * 0.85
        import random
        if random.random() < 0.003:
            self._vx += random.uniform(-2, 2)
        return (float(self._x), float(self._y))

# ── Kalman tracker ────────────────────────────────────────
class BallTracker:
    def __init__(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        kf.R *= 8
        kf.P *= 200
        kf.Q[2:,2:] *= 2
        self.kf   = kf
        self.ready= False
        self.prev_vy = None

    def update(self, det):
        if det is None:
            if self.ready: self.kf.predict()
            return None, False, (0,0)

        if not self.ready:
            self.kf.x = np.array([[det[0]],[det[1]],[0],[0]], dtype=float)
            self.ready = True

        self.kf.predict()
        self.kf.update(np.array([[det[0]],[det[1]]], dtype=float))
        x,y,vx,vy = self.kf.x.flatten()

        bounce = False
        if self.prev_vy is not None and self.prev_vy > 2.0 and vy < 0:
            bounce = True
        self.prev_vy = float(vy)

        return (float(x), float(y)), bounce, (float(vx), float(vy))

# ── Court calibration + line call ─────────────────────────
class Court:
    # Real pickleball court: 6.1m × 13.41m, kitchen = 2.235m from net
    W, H, K = 6.10, 13.41, 2.235

    COURT_POLY  = np.array([[0,0],[6.10,0],[6.10,13.41],[0,13.41]], dtype=np.float32)
    NVZ_NEAR    = np.array([[0,13.41-2.235],[6.10,13.41-2.235],[6.10,13.41],[0,13.41]], dtype=np.float32)
    NVZ_FAR     = np.array([[0,0],[6.10,0],[6.10,2.235],[0,2.235]], dtype=np.float32)

    def __init__(self):
        self.H_mat = None  # homography matrix

    @property
    def ready(self):
        return self.H_mat is not None

    def calibrate(self, src_points):
        """src_points: [[x,y]×4] pixel corners TL→TR→BR→BL"""
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]], dtype=np.float32)
        self.H_mat, _ = cv2.findHomography(src, dst)
        return self.H_mat is not None

    def to_court(self, px, py):
        pt  = np.array([[[px, py]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call(self, cx, cy):
        """Returns dict with result IN/OUT and distance in cm."""
        dist = float(cv2.pointPolygonTest(self.COURT_POLY, (cx, cy), True))
        result = "IN" if dist >= -0.036 else "OUT"  # 3.6cm = ball radius
        return {
            "result":   result,
            "dist_cm":  round(abs(dist) * 100, 1),
            "court_x":  round(cx, 3),
            "court_y":  round(cy, 3),
        }

    def kitchen_zone(self, cx, cy):
        if cv2.pointPolygonTest(self.NVZ_NEAR, (cx, cy), False) >= 0: return "near"
        if cv2.pointPolygonTest(self.NVZ_FAR,  (cx, cy), False) >= 0: return "far"
        return None

# ── Speed calculator ──────────────────────────────────────
class SpeedCalc:
    def __init__(self, fps=30):
        self.fps  = fps
        self.prev = None

    def update(self, court_pos, bounced):
        if bounced: self.prev = None
        if court_pos is None or self.prev is None:
            self.prev = court_pos
            return None
        dx = court_pos[0] - self.prev[0]
        dy = court_pos[1] - self.prev[1]
        dist_m   = (dx**2 + dy**2) ** 0.5
        speed_kmh= dist_m * self.fps * 3.6
        self.prev = court_pos
        if speed_kmh > 140 or speed_kmh < 1: return None
        return round(speed_kmh, 1)

# ── Score state machine ───────────────────────────────────
class Score:
    def __init__(self, doubles=True, win=11):
        self.score   = [0, 0]
        self.serving = 0
        self.server  = 1
        self.doubles = doubles
        self.win     = win
        self.over    = False

    def point(self, team):
        if self.over: return None
        if team == self.serving:
            self.score[team] += 1
            ev = "point"
        else:
            if self.doubles and self.server == 1:
                self.server = 2
            else:
                self.server  = 1
                self.serving = 1 - self.serving
            ev = "sideout"
        s = self.score
        if max(s) >= self.win and abs(s[0]-s[1]) >= 2:
            self.over = True
        return {"event": ev, "score": self.score[:], "serving": self.serving,
                "server": self.server, "game_over": self.over}

    def to_dict(self):
        return {"score": self.score[:], "serving": self.serving,
                "server": self.server, "game_over": self.over}

# ── Shared detector instance (loaded once) ────────────────
detector = BallDetector()

# ── WebSocket endpoint ────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    tracker   = BallTracker()
    court     = Court()
    speed_calc= SpeedCalc(fps=30)
    score     = Score()
    trail     = []

    async def send(data):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── Calibrate ─────────────────────────────────
            if msg.get("type") == "calibrate":
                ok = court.calibrate(msg["points"])
                await send({"type": "calibrated", "ok": ok})
                continue

            # ── Settings ──────────────────────────────────
            if msg.get("type") == "settings":
                score = Score(
                    doubles=msg.get("doubles", True),
                    win=msg.get("win_score", 11)
                )
                await send({"type": "settings_ok"})
                continue

            # ── Reset ─────────────────────────────────────
            if msg.get("type") == "reset":
                tracker    = BallTracker()
                court      = Court()
                speed_calc = SpeedCalc(fps=30)
                score      = Score()
                trail      = []
                await send({"type": "reset_ok"})
                continue

            # ── Frame ─────────────────────────────────────
            if msg.get("type") != "frame":
                continue

            # Decode JPEG
            try:
                img_bytes = base64.b64decode(msg["data"])
                arr   = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            # Detect + track
            det             = detector.detect(frame)
            pos, bounce, vel= tracker.update(det)

            # Trail
            if pos:
                trail.append(list(pos))
                if len(trail) > 24: trail.pop(0)

            # Court-space position + line call
            court_pos    = None
            line_verdict = None
            if court.ready and pos:
                try:
                    cx, cy    = court.to_court(*pos)
                    court_pos = (cx, cy)
                    if bounce:
                        line_verdict = court.call(cx, cy)
                        # Award point on OUT
                        if line_verdict["result"] == "OUT":
                            ev = score.point(1 - score.serving)
                            if ev: line_verdict["score_event"] = ev
                except Exception:
                    pass

            # Speed
            speed = speed_calc.update(court_pos, bounce)

            # Kitchen
            kitchen = None
            if court_pos:
                kitchen = court.kitchen_zone(*court_pos)

            await send({
                "type":       "state",
                "ball":       list(pos) if pos else None,
                "trail":      trail[-24:],
                "bounce":     bounce,
                "verdict":    line_verdict,
                "speed":      speed,
                "kitchen":    kitchen,
                "calibrated": court.ready,
                "score":      score.to_dict(),
            })

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.error(f"Error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
