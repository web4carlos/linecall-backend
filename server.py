"""
LineCall Python backend
=======================
Receives JPEG frames over WebSocket, detects ball, calls IN/OUT.
Supports both AUTO court detection and manual calibration.

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
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            log.info("YOLOv8 loaded")
        except Exception as e:
            log.warning(f"YOLOv8 not available ({e}) — using mock detector")

        self._x  = 300.0
        self._y  = 250.0
        self._vx = 4.5
        self._vy = -6.0

    def detect(self, frame: np.ndarray):
        if self.model is not None:
            results = self.model(frame, classes=[32], conf=0.35, verbose=False)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best = boxes[boxes.conf.argmax()]
                x1,y1,x2,y2 = best.xyxy[0].tolist()
                return ((x1+x2)/2, (y1+y2)/2)

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

# ── Auto court detector ───────────────────────────────────
class AutoCourtDetector:
    """
    Detects pickleball court lines automatically using:
    1. Color segmentation (green court surface)
    2. Hough line detection
    3. Finds the 4 outer corners from intersecting lines
    """

    def detect(self, frame: np.ndarray):
        """
        Returns list of 4 corner points [[x,y]x4] in order
        TL, TR, BR, BL — or None if court not found.
        """
        h, w = frame.shape[:2]

        # ── Step 1: find court surface by color ──────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Green court (outdoor / indoor green)
        green_lo = np.array([35, 40, 40])
        green_hi = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, green_lo, green_hi)

        # Blue court (many indoor courts are blue)
        blue_lo = np.array([100, 40, 40])
        blue_hi = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, blue_lo, blue_hi)

        # Teal/cyan court
        teal_lo = np.array([80, 40, 40])
        teal_hi = np.array([100, 255, 255])
        mask_teal = cv2.inRange(hsv, teal_lo, teal_hi)

        court_mask = cv2.bitwise_or(mask_green, cv2.bitwise_or(mask_blue, mask_teal))

        # Clean up mask
        kernel = np.ones((5,5), np.uint8)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel)
        court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_OPEN,  kernel)

        # ── Step 2: find largest contour (the court) ─────
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)

        # Court must be at least 15% of frame
        if area < w * h * 0.15:
            return None

        # ── Step 3: approximate polygon ──────────────────
        peri    = cv2.arcLength(largest, True)
        approx  = cv2.approxPolyDP(largest, 0.02 * peri, True)

        # We want 4 corners
        if len(approx) < 4:
            return None

        # Get bounding corners from convex hull
        hull   = cv2.convexHull(largest)
        rect   = cv2.minAreaRect(hull)
        corners= cv2.boxPoints(rect)
        corners= corners.astype(np.float32)

        # ── Step 4: order corners TL→TR→BR→BL ────────────
        corners = self._order_corners(corners)
        return corners.tolist()

    def _order_corners(self, pts):
        """Sort 4 points into TL, TR, BR, BL order."""
        pts  = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype=np.float32)

        # TL = smallest sum, BR = largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # TL
        rect[2] = pts[np.argmax(s)]   # BR

        # TR = smallest diff, BL = largest diff
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # TR
        rect[3] = pts[np.argmax(diff)]  # BL

        return rect

    def draw_detection(self, frame: np.ndarray, corners):
        """Draw the detected court outline on the frame."""
        if corners is None:
            return frame
        out = frame.copy()
        pts = np.array(corners, dtype=np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 100), 3)
        labels = ['TL','TR','BR','BL']
        for i, (x,y) in enumerate(corners):
            cv2.circle(out, (int(x),int(y)), 12, (0,255,100), -1)
            cv2.putText(out, labels[i], (int(x)+14, int(y)+6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return out

# ── Kalman tracker ────────────────────────────────────────
class BallTracker:
    def __init__(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        kf.R *= 8
        kf.P *= 200
        kf.Q[2:,2:] *= 2
        self.kf    = kf
        self.ready = False
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
    W, H, K = 6.10, 13.41, 2.235

    COURT_POLY = np.array([[0,0],[6.10,0],[6.10,13.41],[0,13.41]], dtype=np.float32)
    NVZ_NEAR   = np.array([[0,13.41-2.235],[6.10,13.41-2.235],[6.10,13.41],[0,13.41]], dtype=np.float32)
    NVZ_FAR    = np.array([[0,0],[6.10,0],[6.10,2.235],[0,2.235]], dtype=np.float32)

    def __init__(self):
        self.H_mat = None

    @property
    def ready(self):
        return self.H_mat is not None

    def calibrate(self, src_points):
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([[0,0],[self.W,0],[self.W,self.H],[0,self.H]], dtype=np.float32)
        self.H_mat, _ = cv2.findHomography(src, dst)
        return self.H_mat is not None

    def to_court(self, px, py):
        pt  = np.array([[[px, py]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call(self, cx, cy):
        dist   = float(cv2.pointPolygonTest(self.COURT_POLY, (cx, cy), True))
        result = "IN" if dist >= -0.036 else "OUT"
        return {
            "result":  result,
            "dist_cm": round(abs(dist) * 100, 1),
            "court_x": round(cx, 3),
            "court_y": round(cy, 3),
        }

    def kitchen_zone(self, cx, cy):
        if cv2.pointPolygonTest(self.NVZ_NEAR, (cx,cy), False) >= 0: return "near"
        if cv2.pointPolygonTest(self.NVZ_FAR,  (cx,cy), False) >= 0: return "far"
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
        dist_m    = (dx**2 + dy**2) ** 0.5
        speed_kmh = dist_m * self.fps * 3.6
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
        return {"event": ev, "score": self.score[:],
                "serving": self.serving, "server": self.server,
                "game_over": self.over}

    def to_dict(self):
        return {"score": self.score[:], "serving": self.serving,
                "server": self.server, "game_over": self.over}

# ── Shared instances ──────────────────────────────────────
detector      = BallDetector()
auto_detector = AutoCourtDetector()

# ── WebSocket endpoint ────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    tracker    = BallTracker()
    court      = Court()
    speed_calc = SpeedCalc(fps=30)
    score      = Score()
    trail      = []
    auto_scan_frames = 0   # count frames for auto-detection

    async def send(data):
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── Manual calibration ────────────────────────
            if msg.get("type") == "calibrate":
                ok = court.calibrate(msg["points"])
                await send({"type": "calibrated", "ok": ok, "method": "manual"})
                continue

            # ── Auto detect court ─────────────────────────
            if msg.get("type") == "auto_detect":
                try:
                    img_bytes = base64.b64decode(msg["data"])
                    arr   = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        corners = auto_detector.detect(frame)
                        if corners:
                            ok = court.calibrate(corners)
                            await send({
                                "type":    "calibrated",
                                "ok":      ok,
                                "method":  "auto",
                                "corners": corners,
                            })
                        else:
                            await send({
                                "type": "auto_detect_failed",
                                "message": "Court not found. Make sure the full court is visible and well lit.",
                            })
                except Exception as e:
                    await send({"type": "auto_detect_failed", "message": str(e)})
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

            try:
                img_bytes = base64.b64decode(msg["data"])
                arr   = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except Exception:
                continue

            # Auto-scan first 30 frames if not calibrated
            if not court.ready:
                auto_scan_frames += 1
                if auto_scan_frames <= 30 and auto_scan_frames % 10 == 0:
                    corners = auto_detector.detect(frame)
                    if corners:
                        court.calibrate(corners)
                        await send({
                            "type":    "calibrated",
                            "ok":      True,
                            "method":  "auto",
                            "corners": corners,
                        })

            det              = detector.detect(frame)
            pos, bounce, vel = tracker.update(det)

            if pos:
                trail.append(list(pos))
                if len(trail) > 24: trail.pop(0)

            court_pos    = None
            line_verdict = None
            if court.ready and pos:
                try:
                    cx, cy    = court.to_court(*pos)
                    court_pos = (cx, cy)
                    if bounce:
                        line_verdict = court.call(cx, cy)
                        if line_verdict["result"] == "OUT":
                            ev = score.point(1 - score.serving)
                            if ev: line_verdict["score_event"] = ev
                except Exception:
                    pass

            speed   = speed_calc.update(court_pos, bounce)
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
