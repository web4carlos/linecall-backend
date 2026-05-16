"""
LineCall backend v3 — reescrito para mayor precisión
- Detección de bounce más robusta (umbral de velocidad vertical)
- Calibración manual mejorada con zoom en las esquinas
- Auto-detect simplificado y más conservador
- OUT solo se llama cuando el bounce es claro y fuera de líneas
"""

import asyncio, base64, json, logging, os
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("linecall")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Ball detector ─────────────────────────────────────────
class BallDetector:
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            log.info("YOLOv8 loaded")
        except Exception as e:
            log.warning(f"YOLOv8 not available: {e}")

    def detect(self, frame: np.ndarray):
        """Returns (cx, cy, confidence) or None."""
        if self.model is not None:
            results = self.model(frame, classes=[32], conf=0.3, verbose=False)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best = boxes[boxes.conf.argmax()]
                x1,y1,x2,y2 = best.xyxy[0].tolist()
                conf = float(best.conf[0])
                return ((x1+x2)/2, (y1+y2)/2, conf)
        return None

# ── Ball tracker with robust bounce detection ─────────────
class BallTracker:
    """
    Tracks ball position and detects bounces.
    Bounce = ball moving DOWN then suddenly moves UP (vy sign change)
    Only triggers if vertical speed change is significant.
    """
    def __init__(self):
        self.positions  = []   # last N (x, y) positions
        self.MAX_HIST   = 8
        self.last_vy    = None
        self.bounce_cooldown = 0  # frames to wait before next bounce

    def update(self, det):
        """Returns (smoothed_pos, is_bounce) or (None, False)."""
        if det is None:
            self.bounce_cooldown = max(0, self.bounce_cooldown - 1)
            return None, False

        x, y, conf = det
        self.positions.append((x, y))
        if len(self.positions) > self.MAX_HIST:
            self.positions.pop(0)

        # Need at least 4 points to estimate velocity
        if len(self.positions) < 4:
            return (x, y), False

        # Smooth position (average of last 3)
        recent = self.positions[-3:]
        sx = sum(p[0] for p in recent) / len(recent)
        sy = sum(p[1] for p in recent) / len(recent)

        # Vertical velocity (positive = moving down in image coords)
        n   = len(self.positions)
        vy  = self.positions[-1][1] - self.positions[-3][1]

        bounce = False
        if (self.last_vy is not None
                and self.last_vy > 6       # was moving down fast
                and vy < -4                # now moving up
                and self.bounce_cooldown == 0):
            bounce = True
            self.bounce_cooldown = 12  # don't trigger again for 12 frames (~0.4s)
            log.info(f"Bounce detected at ({sx:.0f},{sy:.0f}) vy: {self.last_vy:.1f}→{vy:.1f}")

        self.last_vy = float(vy)
        self.bounce_cooldown = max(0, self.bounce_cooldown - 1)
        return (float(sx), float(sy)), bounce

    def reset(self):
        self.positions = []
        self.last_vy   = None
        self.bounce_cooldown = 0

# ── Court ─────────────────────────────────────────────────
class Court:
    """
    Real pickleball court dimensions:
    - Total: 6.10m wide x 13.41m long
    - NVZ (kitchen): 2.235m from net each side
    - Net at 6.705m from each baseline
    """
    W  = 6.10
    H  = 13.41
    NVZ= 2.235

    def __init__(self):
        self.H_mat    = None
        self.src_pts  = None  # original pixel corners

        # Court polygon in metres
        self.court_poly = np.array([
            [0, 0], [self.W, 0], [self.W, self.H], [0, self.H]
        ], dtype=np.float32)

        # Kitchen polygons
        self.nvz_bottom = np.array([
            [0, self.H - self.NVZ], [self.W, self.H - self.NVZ],
            [self.W, self.H],        [0, self.H]
        ], dtype=np.float32)
        self.nvz_top = np.array([
            [0, 0], [self.W, 0],
            [self.W, self.NVZ], [0, self.NVZ]
        ], dtype=np.float32)

    @property
    def ready(self):
        return self.H_mat is not None

    def calibrate(self, src_points):
        """
        src_points: [[x,y]×4] pixel corners in order TL→TR→BR→BL
        Maps to real court coordinates.
        """
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([
            [0,        0       ],
            [self.W,   0       ],
            [self.W,   self.H  ],
            [0,        self.H  ],
        ], dtype=np.float32)
        self.H_mat, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        self.src_pts = src
        ok = self.H_mat is not None
        if ok:
            log.info(f"Court calibrated. Corners: {src_points}")
        return ok

    def to_court(self, px, py):
        """Convert pixel (px,py) → court metres (cx,cy)."""
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call(self, cx, cy):
        """
        IN/OUT verdict.
        Returns dict with result, distance in cm, court coords.
        Tolerance: 3.6cm (ball radius) — ball is IN if centre is within tolerance.
        """
        # Distance from court boundary (positive=inside, negative=outside)
        dist_m = float(cv2.pointPolygonTest(
            self.court_poly, (cx, cy), measureDist=True
        ))
        # Ball is IN if centre is inside, or within 3.6cm of line
        BALL_RADIUS_M = 0.036
        result = "IN" if dist_m >= -BALL_RADIUS_M else "OUT"

        return {
            "result":  result,
            "dist_cm": round(abs(dist_m) * 100, 1),
            "court_x": round(cx, 3),
            "court_y": round(cy, 3),
        }

    def in_kitchen(self, cx, cy):
        if cv2.pointPolygonTest(self.nvz_bottom, (cx, cy), False) >= 0: return "bottom"
        if cv2.pointPolygonTest(self.nvz_top,    (cx, cy), False) >= 0: return "top"
        return None

# ── Auto court detector ───────────────────────────────────
class AutoCourtDetector:
    """
    Conservative court detector.
    Tries multiple color ranges and validates aspect ratio.
    Returns None if not confident enough.
    """
    COLOR_RANGES = [
        # Green
        (np.array([30, 30, 40]),  np.array([90, 255, 255])),
        # Blue
        (np.array([95, 40, 30]),  np.array([135, 255, 255])),
        # Teal
        (np.array([78, 30, 30]),  np.array([102, 255, 255])),
        # Dark blue (compressed video)
        (np.array([90, 15, 15]),  np.array([140, 180, 200])),
    ]

    def detect(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        min_area = w * h * 0.18  # court must be at least 18% of frame
        max_area = w * h * 0.92

        best_corners = None
        best_score   = 0

        for lo, hi in self.COLOR_RANGES:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)

            k    = np.ones((11, 11), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue

            largest = max(cnts, key=cv2.contourArea)
            area    = cv2.contourArea(largest)
            if area < min_area or area > max_area: continue

            hull  = cv2.convexHull(largest)
            rect  = cv2.minAreaRect(hull)
            box   = cv2.boxPoints(rect).astype(np.float32)

            # Check aspect ratio
            s0 = np.linalg.norm(box[0]-box[1])
            s1 = np.linalg.norm(box[1]-box[2])
            if s0 < 1 or s1 < 1: continue
            ratio = max(s0,s1) / min(s0,s1)
            if ratio < 1.2 or ratio > 6.0: continue

            score = area  # bigger = better
            if score > best_score:
                best_score   = score
                best_corners = self._order(box).tolist()

        return best_corners

    def _order(self, pts):
        pts  = pts.reshape(4,2)
        rect = np.zeros((4,2), dtype=np.float32)
        s    = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d    = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

# ── Speed calculator ──────────────────────────────────────
class SpeedCalc:
    def __init__(self, fps=25):
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
        if speed_kmh > 160 or speed_kmh < 2: return None
        return round(speed_kmh, 1)

# ── Score machine ─────────────────────────────────────────
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
        return {"event":ev, "score":self.score[:],
                "serving":self.serving, "server":self.server, "game_over":self.over}

    def to_dict(self):
        return {"score":self.score[:], "serving":self.serving,
                "server":self.server, "game_over":self.over}

# ── Instances ─────────────────────────────────────────────
detector      = BallDetector()
auto_detector = AutoCourtDetector()

# ── WebSocket ─────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected")

    tracker    = BallTracker()
    court      = Court()
    speed_calc = SpeedCalc(fps=25)
    score      = Score()
    trail      = []
    auto_frames= 0

    async def send(data):
        try: await ws.send_text(json.dumps(data))
        except: pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── Calibrate manually ────────────────────────
            if msg.get("type") == "calibrate":
                ok = court.calibrate(msg["points"])
                tracker.reset()
                await send({"type":"calibrated","ok":ok,"method":"manual"})
                continue

            # ── Auto detect ───────────────────────────────
            if msg.get("type") == "auto_detect":
                try:
                    img_bytes = base64.b64decode(msg["data"])
                    arr   = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        corners = auto_detector.detect(frame)
                        # Try with contrast enhancement if failed
                        if not corners:
                            enh     = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
                            corners = auto_detector.detect(enh)
                        if corners:
                            ok = court.calibrate(corners)
                            tracker.reset()
                            await send({
                                "type":"calibrated","ok":ok,
                                "method":"auto","corners":corners
                            })
                        else:
                            await send({
                                "type":"auto_detect_failed",
                                "message":
                                    "No se detectó la cancha.\n\n"
                                    "Consejos:\n"
                                    "• Pausa el video en un frame donde se vea la cancha completa\n"
                                    "• La cancha debe ocupar al menos 20% de la imagen\n"
                                    "• Usa Calibración Manual si el auto falla\n"
                                    "• Asegúrate que haya buena iluminación"
                            })
                except Exception as e:
                    await send({"type":"auto_detect_failed","message":str(e)})
                continue

            # ── Settings ──────────────────────────────────
            if msg.get("type") == "settings":
                score = Score(doubles=msg.get("doubles",True), win=msg.get("win_score",11))
                await send({"type":"settings_ok"})
                continue

            # ── Reset ─────────────────────────────────────
            if msg.get("type") == "reset":
                tracker    = BallTracker()
                court      = Court()
                speed_calc = SpeedCalc(fps=25)
                score      = Score()
                trail      = []
                auto_frames= 0
                await send({"type":"reset_ok"})
                continue

            # ── Frame ─────────────────────────────────────
            if msg.get("type") != "frame":
                continue

            try:
                img_bytes = base64.b64decode(msg["data"])
                arr   = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except:
                continue

            # Auto-scan first 90 frames (every 15)
            if not court.ready:
                auto_frames += 1
                if auto_frames % 15 == 0 and auto_frames <= 90:
                    corners = auto_detector.detect(frame)
                    if corners:
                        court.calibrate(corners)
                        tracker.reset()
                        await send({
                            "type":"calibrated","ok":True,
                            "method":"auto","corners":corners
                        })

            # Detect ball
            det = detector.detect(frame)

            # Track + bounce
            pos, bounce = tracker.update(det)

            # Trail
            if pos:
                trail.append(list(pos))
                if len(trail) > 30: trail.pop(0)

            # Line call — only on bounce
            court_pos    = None
            line_verdict = None

            if court.ready and pos:
                try:
                    cx, cy    = court.to_court(*pos)
                    court_pos = (cx, cy)

                    if bounce:
                        verdict = court.call(cx, cy)
                        line_verdict = verdict

                        # Only award point on clear OUT
                        if verdict["result"] == "OUT":
                            ev = score.point(1 - score.serving)
                            if ev: line_verdict["score_event"] = ev

                        log.info(
                            f"Bounce → {verdict['result']} "
                            f"court=({cx:.2f},{cy:.2f}) "
                            f"dist={verdict['dist_cm']}cm"
                        )
                except Exception as e:
                    log.error(f"Court transform error: {e}")

            speed   = speed_calc.update(court_pos, bounce)
            kitchen = court.in_kitchen(*court_pos) if court_pos else None

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
        log.error(f"Session error: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
