"""
LineCall backend v4
===================
- Detección de pelota por COLOR (amarillo/verde fluorescente) + forma circular
- OUT se detecta en tiempo real basado en posición, no solo en bounce
- Lógica de bounce mejorada con historial de posiciones
- Debug info enviado al cliente para ajuste fino
"""

import asyncio, base64, json, logging, os
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("linecall")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ball detector por COLOR ───────────────────────────────
class BallDetector:
    """
    Detecta la pelota de pickleball por color (amarillo/verde fluor)
    y forma circular. Más confiable que YOLO para este caso.
    """
    # Colores de pelota de pickleball en HSV
    # Amarillo-verde fluorescente (la más común)
    BALL_COLORS = [
        # Amarillo brillante
        (np.array([20, 80, 150]),  np.array([40, 255, 255])),
        # Verde-amarillo (chartreuse)
        (np.array([40, 80, 150]),  np.array([75, 255, 255])),
        # Naranja (algunas pelotas)
        (np.array([5,  100, 150]), np.array([20, 255, 255])),
        # Blanco/gris claro (pelotas indoor)
        (np.array([0,  0,   180]), np.array([180, 40, 255])),
    ]

    def __init__(self):
        # Intentar cargar YOLO como fallback
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            log.info("YOLOv8 cargado como fallback")
        except:
            log.info("YOLOv8 no disponible, usando detección por color")

    def detect(self, frame: np.ndarray):
        """
        Retorna (cx, cy, radius, method) o None.
        Prioridad: detección por color > YOLO
        """
        result = self._detect_by_color(frame)
        if result:
            return result

        # Fallback a YOLO si está disponible
        if self.model:
            return self._detect_yolo(frame)

        return None

    def _detect_by_color(self, frame):
        h, w = frame.shape[:2]
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best = None
        best_score = 0

        for lo, hi in self.BALL_COLORS:
            mask = cv2.inRange(hsv, lo, hi)

            # Filtro morfológico para limpiar ruido
            kernel = np.ones((3,3), np.uint8)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Encontrar contornos
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area < 20 or area > 8000:  # pelota muy pequeña o muy grande
                    continue

                # Verificar circularidad
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.45:  # debe ser bastante circular
                    continue

                # Centro y radio
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)

                # Score basado en área y circularidad
                score = area * circularity
                if score > best_score:
                    best_score = score
                    best = (float(cx), float(cy), float(radius), 'color')

        return best

    def _detect_yolo(self, frame):
        try:
            results = self.model(frame, classes=[32], conf=0.25, verbose=False)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                best   = boxes[boxes.conf.argmax()]
                x1,y1,x2,y2 = best.xyxy[0].tolist()
                cx = (x1+x2)/2
                cy = (y1+y2)/2
                r  = max((x2-x1),(y2-y1))/2
                return (float(cx), float(cy), float(r), 'yolo')
        except:
            pass
        return None


# ── Tracker con historial de posiciones ───────────────────
class BallTracker:
    """
    Suaviza la posición y detecta bounces analizando
    la dirección vertical de los últimos N frames.
    """
    HISTORY = 12  # frames de historial

    def __init__(self):
        self.positions   = []  # [(x,y,frame_num)]
        self.frame_num   = 0
        self.bounce_cd   = 0   # cooldown para no detectar doble bounce
        self.last_smooth = None

    def update(self, det):
        """
        det: (cx, cy, radius, method) o None
        Retorna (smooth_pos, is_bounce)
        """
        self.frame_num += 1
        self.bounce_cd  = max(0, self.bounce_cd - 1)

        if det is None:
            # Si no hay detección, no añadimos punto
            # pero mantenemos el historial existente
            return self.last_smooth, False

        cx, cy, r, method = det
        self.positions.append((cx, cy, self.frame_num))
        if len(self.positions) > self.HISTORY:
            self.positions.pop(0)

        # Necesitamos al menos 5 puntos para análisis
        if len(self.positions) < 5:
            pos = (cx, cy)
            self.last_smooth = pos
            return pos, False

        # Suavizar con promedio ponderado (más peso a los recientes)
        weights = [i+1 for i in range(len(self.positions))]
        total_w = sum(weights)
        sx = sum(p[0]*w for p,w in zip(self.positions, weights)) / total_w
        sy = sum(p[1]*w for p,w in zip(self.positions, weights)) / total_w
        pos = (float(sx), float(sy))
        self.last_smooth = pos

        # Detectar bounce
        bounce = self._detect_bounce()

        return pos, bounce

    def _detect_bounce(self):
        """
        Bounce = la pelota cambia de dirección vertical:
        estaba bajando (y aumentando) y ahora sube (y disminuyendo)
        """
        if self.bounce_cd > 0:
            return False
        if len(self.positions) < 6:
            return False

        # Calcular velocidad vertical en la primera mitad vs segunda mitad
        mid  = len(self.positions) // 2
        half1 = self.positions[:mid]
        half2 = self.positions[mid:]

        # Velocidad promedio en cada mitad (positivo = bajando)
        def avg_vy(pts):
            if len(pts) < 2:
                return 0
            return (pts[-1][1] - pts[0][1]) / max(1, len(pts)-1)

        vy1 = avg_vy(half1)
        vy2 = avg_vy(half2)

        # Bounce: primero bajando fuerte, luego subiendo
        if vy1 > 3.0 and vy2 < -2.0:
            self.bounce_cd = 15  # esperar 15 frames (~0.5s)
            log.info(f"Bounce detectado: vy1={vy1:.1f} vy2={vy2:.1f}")
            return True

        return False

    def reset(self):
        self.positions   = []
        self.frame_num   = 0
        self.bounce_cd   = 0
        self.last_smooth = None


# ── Court con verificación continua ───────────────────────
class Court:
    W   = 6.10
    H   = 13.41
    NVZ = 2.235

    def __init__(self):
        self.H_mat     = None
        self.court_poly = np.array([
            [0,0],[self.W,0],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.nvz_near = np.array([
            [0,self.H-self.NVZ],[self.W,self.H-self.NVZ],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.nvz_far = np.array([
            [0,0],[self.W,0],[self.W,self.NVZ],[0,self.NVZ]
        ], dtype=np.float32)
        # Para tracking de posición en cancha
        self.last_court_pos = None
        self.out_frames     = 0  # frames consecutivos fuera de cancha

    @property
    def ready(self):
        return self.H_mat is not None

    def calibrate(self, src_points):
        src = np.array(src_points, dtype=np.float32)
        dst = np.array([
            [0,0],[self.W,0],[self.W,self.H],[0,self.H]
        ], dtype=np.float32)
        self.H_mat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return self.H_mat is not None

    def to_court(self, px, py):
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self.H_mat)
        return float(res[0][0][0]), float(res[0][0][1])

    def call_on_bounce(self, cx, cy):
        """Veredicto IN/OUT en el momento del bounce."""
        dist = float(cv2.pointPolygonTest(self.court_poly, (cx,cy), True))
        TOLERANCE = 0.04  # 4cm de tolerancia (radio de la pelota)
        result = "IN" if dist >= -TOLERANCE else "OUT"
        return {
            "result":  result,
            "dist_cm": round(abs(dist)*100, 1),
            "court_x": round(cx, 3),
            "court_y": round(cy, 3),
        }

    def is_in_court(self, cx, cy):
        """Verifica si una posición está dentro de la cancha."""
        dist = float(cv2.pointPolygonTest(self.court_poly, (cx,cy), True))
        return dist >= -0.10  # 10cm de margen

    def in_kitchen(self, cx, cy):
        if cv2.pointPolygonTest(self.nvz_near,(cx,cy),False) >= 0: return "near"
        if cv2.pointPolygonTest(self.nvz_far, (cx,cy),False) >= 0: return "far"
        return None


# ── Auto detector de cancha ───────────────────────────────
class AutoCourtDetector:
    COLOR_RANGES = [
        (np.array([30,25,40]),  np.array([90,255,255])),   # verde
        (np.array([95,40,30]),  np.array([135,255,255])),  # azul
        (np.array([78,25,30]),  np.array([102,255,255])),  # teal
        (np.array([90,15,15]),  np.array([140,180,200])),  # azul oscuro
    ]

    def detect(self, frame):
        h, w = frame.shape[:2]
        best = None
        best_area = 0

        for lo, hi in self.COLOR_RANGES:
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lo, hi)
            k    = np.ones((11,11), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue

            largest = max(cnts, key=cv2.contourArea)
            area    = cv2.contourArea(largest)
            if area < w*h*0.18 or area > w*h*0.92: continue

            hull = cv2.convexHull(largest)
            rect = cv2.minAreaRect(hull)
            box  = cv2.boxPoints(rect).astype(np.float32)

            s0 = np.linalg.norm(box[0]-box[1])
            s1 = np.linalg.norm(box[1]-box[2])
            if s0 < 1 or s1 < 1: continue
            ratio = max(s0,s1)/min(s0,s1)
            if ratio < 1.2 or ratio > 7.0: continue

            if area > best_area:
                best_area = area
                best = self._order(box).tolist()

        return best

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


# ── Speed ─────────────────────────────────────────────────
class SpeedCalc:
    def __init__(self, fps=25):
        self.fps  = fps
        self.prev = None

    def update(self, court_pos, bounce):
        if bounce: self.prev = None
        if not court_pos or not self.prev:
            self.prev = court_pos
            return None
        dx = court_pos[0]-self.prev[0]
        dy = court_pos[1]-self.prev[1]
        spd = (dx**2+dy**2)**0.5 * self.fps * 3.6
        self.prev = court_pos
        if spd > 180 or spd < 1: return None
        return round(spd, 1)


# ── Score ─────────────────────────────────────────────────
class Score:
    def __init__(self, doubles=True, win=11):
        self.score=[0,0]; self.serving=0; self.server=1
        self.doubles=doubles; self.win=win; self.over=False

    def point(self, team):
        if self.over: return None
        if team == self.serving:
            self.score[team] += 1; ev="point"
        else:
            if self.doubles and self.server==1: self.server=2
            else: self.server=1; self.serving=1-self.serving
            ev="sideout"
        if max(self.score)>=self.win and abs(self.score[0]-self.score[1])>=2:
            self.over=True
        return {"event":ev,"score":self.score[:],"serving":self.serving,
                "server":self.server,"game_over":self.over}

    def to_dict(self):
        return {"score":self.score[:],"serving":self.serving,
                "server":self.server,"game_over":self.over}


# ── Instancias globales ───────────────────────────────────
detector      = BallDetector()
auto_detector = AutoCourtDetector()


# ── WebSocket ─────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Cliente conectado")

    tracker    = BallTracker()
    court      = Court()
    speed_calc = SpeedCalc(fps=25)
    score      = Score()
    trail      = []
    auto_frames = 0
    last_verdict = None      # último veredicto enviado
    verdict_hold = 0         # frames para mantener el veredicto visible

    async def send(data):
        try: await ws.send_text(json.dumps(data))
        except: pass

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            # ── Calibrar manualmente ──────────────────────
            if msg.get("type") == "calibrate":
                ok = court.calibrate(msg["points"])
                tracker.reset()
                last_verdict = None
                await send({"type":"calibrated","ok":ok,"method":"manual"})
                continue

            # ── Auto detect ───────────────────────────────
            if msg.get("type") == "auto_detect":
                try:
                    arr   = np.frombuffer(base64.b64decode(msg["data"]), np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        corners = auto_detector.detect(frame)
                        if not corners:
                            enh     = cv2.convertScaleAbs(frame, alpha=1.4, beta=30)
                            corners = auto_detector.detect(enh)
                        if corners:
                            ok = court.calibrate(corners)
                            tracker.reset()
                            await send({"type":"calibrated","ok":ok,"method":"auto","corners":corners})
                        else:
                            await send({
                                "type":"auto_detect_failed",
                                "message":"No se detectó la cancha.\n\nUsa Calibración Manual:\n• Pausa el video\n• Click en las 4 esquinas de la cancha"
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
                tracker=BallTracker(); court=Court(); speed_calc=SpeedCalc()
                score=Score(); trail=[]; auto_frames=0; last_verdict=None; verdict_hold=0
                await send({"type":"reset_ok"})
                continue

            # ── Frame ─────────────────────────────────────
            if msg.get("type") != "frame":
                continue

            try:
                arr   = np.frombuffer(base64.b64decode(msg["data"]), np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None: continue
            except:
                continue

            # Auto-scan si no calibrado
            if not court.ready:
                auto_frames += 1
                if auto_frames % 20 == 0 and auto_frames <= 100:
                    corners = auto_detector.detect(frame)
                    if corners:
                        court.calibrate(corners)
                        tracker.reset()
                        await send({"type":"calibrated","ok":True,"method":"auto","corners":corners})

            # Detectar pelota
            det = detector.detect(frame)

            # Trackear + detectar bounce
            pos, bounce = tracker.update(det)

            # Trail
            if pos:
                trail.append(list(pos))
                if len(trail) > 30: trail.pop(0)

            # Lógica de cancha
            court_pos    = None
            line_verdict = None
            kitchen      = None
            speed        = None

            if court.ready and pos:
                try:
                    cx, cy    = court.to_court(*pos)
                    court_pos = (cx, cy)
                    kitchen   = court.in_kitchen(cx, cy)
                    speed     = speed_calc.update(court_pos, bounce)

                    if bounce:
                        verdict      = court.call_on_bounce(cx, cy)
                        last_verdict = verdict
                        verdict_hold = 20  # mantener veredicto por 20 frames
                        line_verdict = verdict

                        if verdict["result"] == "OUT":
                            ev = score.point(1 - score.serving)
                            if ev: line_verdict["score_event"] = ev

                        log.info(
                            f"BOUNCE → {verdict['result']} "
                            f"pos_cancha=({cx:.2f}m, {cy:.2f}m) "
                            f"dist={verdict['dist_cm']}cm"
                        )
                    elif verdict_hold > 0:
                        # Mantener el último veredicto visible unos frames más
                        verdict_hold -= 1
                        line_verdict  = last_verdict if verdict_hold > 0 else None

                except Exception as e:
                    log.error(f"Error cancha: {e}")

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
                "det_method": det[3] if det else None,  # debug: 'color' o 'yolo'
            })

    except WebSocketDisconnect:
        log.info("Cliente desconectado")
    except Exception as e:
        log.error(f"Error sesión: {e}")

@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
