"""
Detector de Somnolencia para Laptop
=====================================
Compatible con MediaPipe 0.10.x y Python 3.14
Detecta: ojos cerrados + bostezos
Alerta por correo Gmail
 
Requisitos:
    python -m pip install opencv-python mediapipe numpy scipy
 
Uso:
    python detector_somnolencia.py
"""
 
import cv2
import numpy as np
import time
import sys
import urllib.request
import os
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
 
# ==============================================================================
#  CONFIGURACION — edita estos 3 valores antes de correr
# ==============================================================================
GMAIL_ORIGEN   = "bigpatch09@gmail.com"
GMAIL_PASSWORD = "erad ppip opjy rehb"
CORREO_DESTINO = "bigpatch09@gmail.com"
# ==============================================================================
 
COOLDOWN_EMAIL    = 60     # segundos entre correos
 
# ── Constantes ojos (EAR) ─────────────────────────────────────────────────────
EAR_THRESHOLD     = 0.25
EAR_CONSEC_FRAMES = 20
BLINK_CONSEC_MIN  = 2
 
# ── Constantes boca (MAR) ─────────────────────────────────────────────────────
MAR_THRESHOLD     = 0.6    # Por encima → boca muy abierta (bostezo)
MAR_CONSEC_FRAMES = 15     # Frames seguidos con boca abierta → bostezo confirmado
 
# ── Landmarks MediaPipe ───────────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
 
# Boca: 4 puntos verticales + 2 horizontales
# Superior: 13, Inferior: 14, Izq: 78, Der: 308, SupExt: 82, InfExt: 87
MOUTH_IDX = [78, 82, 13, 312, 308, 317, 14, 87]
# p1=izq(78) p2=sup-izq(82) p3=sup-centro(13) p4=der(308)
# p5=inf-der(317) p6=inf-centro(14) p7=inf-izq(87) p8=sup-der(312)
 
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"
 
 
# ── Alerta sonora ─────────────────────────────────────────────────────────────
def make_beep(freq=1000, dur=1.0):
    try:
        import sounddevice as sd
        sample_rate = 44100
        t    = np.linspace(0, dur, int(sample_rate * dur), False)
        wave = (np.sin(2 * np.pi * freq * t) * 0.8).astype(np.float32)
        sd.play(wave, sample_rate)
        sd.wait()
    except Exception:
        try:
            import winsound
            winsound.Beep(freq, int(dur * 1000))
        except Exception:
            print("\a", end="", flush=True)
 
 
# ── Alerta por correo ─────────────────────────────────────────────────────────
def enviar_correo(tipo_alerta, hora_alerta, frame_captura=None):
    try:
        if GMAIL_ORIGEN == "tu_correo@gmail.com":
            print("[EMAIL] Configura GMAIL_ORIGEN y GMAIL_PASSWORD en el codigo.")
            return
 
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_ORIGEN
        msg["To"]      = CORREO_DESTINO
        msg["Subject"] = f"ALERTA: {tipo_alerta} detectado"
 
        cuerpo = f"""
ALERTA DEL DETECTOR DE SOMNOLENCIA
====================================
Tipo de alerta : {tipo_alerta}
Hora           : {hora_alerta}
 
{"Los ojos permanecieron cerrados por demasiado tiempo." if "somnolencia" in tipo_alerta.lower() else "Se detecto un bostezo prolongado."}
 
Por favor detente y descansa.
Se adjunta captura del momento de la alerta.
        """
        msg.attach(MIMEText(cuerpo, "plain"))
 
        # Adjuntar captura si existe
        if frame_captura is not None:
            ret, buffer = cv2.imencode(".jpg", frame_captura, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                img_bytes = buffer.tobytes()
                imagen = MIMEImage(img_bytes, name=f"alerta_{hora_alerta.replace(':','-')}.jpg")
                imagen.add_header("Content-Disposition", "attachment",
                                  filename=f"alerta_{hora_alerta.replace(':','-')}.jpg")
                msg.attach(imagen)
 
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as servidor:
            servidor.login(GMAIL_ORIGEN, GMAIL_PASSWORD)
            servidor.sendmail(GMAIL_ORIGEN, CORREO_DESTINO, msg.as_string())
 
        print(f"[EMAIL] Alerta '{tipo_alerta}' enviada con captura a {CORREO_DESTINO}")
 
    except smtplib.SMTPAuthenticationError:
        print("[EMAIL] Error: contrasena de aplicacion incorrecta.")
    except Exception as e:
        print(f"[EMAIL] Error al enviar: {e}")
 
 
def enviar_correo_async(tipo, hora, frame_captura=None):
    # Copiar el frame para que no cambie mientras se envia
    frame_copia = frame_captura.copy() if frame_captura is not None else None
    threading.Thread(target=enviar_correo, args=(tipo, hora, frame_copia), daemon=True).start()
 
 
# ── Descarga modelo ───────────────────────────────────────────────────────────
def download_model():
    if os.path.exists(MODEL_PATH):
        return
    print("Descargando modelo de deteccion facial (~29 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo descargado.")
 
 
# ── Calculo EAR ───────────────────────────────────────────────────────────────
def calc_ear(landmarks, indices, w, h):
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)
 
 
# ── Calculo MAR (Mouth Aspect Ratio) ─────────────────────────────────────────
def calc_mar(landmarks, w, h):
    # Puntos de la boca
    p = {i: np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in MOUTH_IDX}
    # Distancias verticales
    A = np.linalg.norm(p[82]  - p[87])   # sup-izq a inf-izq
    B = np.linalg.norm(p[13]  - p[14])   # sup-centro a inf-centro
    C = np.linalg.norm(p[312] - p[317])  # sup-der a inf-der
    # Distancia horizontal
    D = np.linalg.norm(p[78]  - p[308])  # izquierda a derecha
    return (A + B + C) / (3.0 * D + 1e-6)
 
 
def draw_contour(frame, landmarks, indices, w, h, color):
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)
 
 
# ── Loop principal ────────────────────────────────────────────────────────────
def run_detector():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError:
        print("ERROR: MediaPipe no esta instalado.")
        sys.exit(1)
 
    download_model()
 
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se puede acceder a la camara.")
        sys.exit(1)
 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    # Estado ojos
    ear_counter   = 0
    blink_counter = 0
    total_blinks  = 0
    alert_ojos    = False
 
    # Estado boca
    mar_counter   = 0
    total_bostezos = 0
    alert_bostezo  = False
 
    ultimo_email  = 0
    start_time    = time.time()
 
    print("Detector iniciado. Presiona 'q' para salir.")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        rgb_np   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
        result   = detector.detect(mp_image)
 
        status_text  = "No se detecta rostro"
        status_color = (0, 165, 255)
        ear_avg = mar_val = 0.0
 
        if result.face_landmarks:
            lms = result.face_landmarks[0]
 
            # ── Deteccion de ojos ─────────────────────────────────────────────
            ear_l   = calc_ear(lms, LEFT_EYE_IDX,  w, h)
            ear_r   = calc_ear(lms, RIGHT_EYE_IDX, w, h)
            ear_avg = (ear_l + ear_r) / 2.0
 
            eye_color = (0, 255, 0) if ear_avg >= EAR_THRESHOLD else (0, 0, 255)
            draw_contour(frame, lms, LEFT_EYE_IDX,  w, h, eye_color)
            draw_contour(frame, lms, RIGHT_EYE_IDX, w, h, eye_color)
 
            if ear_avg < EAR_THRESHOLD:
                ear_counter += 1
                if ear_counter >= BLINK_CONSEC_MIN:
                    blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_MIN:
                    total_blinks += 1
                blink_counter = 0
                if ear_counter >= EAR_CONSEC_FRAMES:
                    alert_ojos = True
                    make_beep(freq=1000, dur=1.0)
                    ahora = time.time()
                    if ahora - ultimo_email >= COOLDOWN_EMAIL:
                        ultimo_email = ahora
                        enviar_correo_async("Somnolencia (ojos cerrados)", time.strftime("%H:%M:%S"), frame)
                ear_counter = 0
                alert_ojos  = False
 
            # ── Deteccion de bostezo ──────────────────────────────────────────
            mar_val    = calc_mar(lms, w, h)
            mouth_color = (0, 255, 255) if mar_val >= MAR_THRESHOLD else (0, 200, 0)
            draw_contour(frame, lms, MOUTH_IDX, w, h, mouth_color)
 
            if mar_val >= MAR_THRESHOLD:
                mar_counter += 1
            else:
                if mar_counter >= MAR_CONSEC_FRAMES:
                    total_bostezos += 1
                    alert_bostezo = True
                    make_beep(freq=800, dur=0.7)
                    ahora = time.time()
                    if ahora - ultimo_email >= COOLDOWN_EMAIL:
                        ultimo_email = ahora
                        enviar_correo_async("Bostezo detectado", time.strftime("%H:%M:%S"), frame)
                else:
                    alert_bostezo = False
                mar_counter = 0
 
            # Estado texto
            if alert_ojos or ear_counter >= EAR_CONSEC_FRAMES:
                status_text  = f"OJOS CERRADOS  EAR: {ear_avg:.2f}"
                status_color = (0, 0, 255)
            elif alert_bostezo or mar_counter >= MAR_CONSEC_FRAMES:
                status_text  = f"BOSTEZO  MAR: {mar_val:.2f}"
                status_color = (0, 200, 255)
            else:
                status_text  = f"Normal  EAR:{ear_avg:.2f}  MAR:{mar_val:.2f}"
                status_color = (0, 200, 0)
 
        # ── Alertas visuales ──────────────────────────────────────────────────
        if alert_ojos or ear_counter >= EAR_CONSEC_FRAMES:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.putText(frame, "*** ALERTA SOMNOLENCIA ***",
                        (w // 2 - 220, h // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
 
        if alert_bostezo or mar_counter >= MAR_CONSEC_FRAMES:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 100, 180), -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.putText(frame, "*** BOSTEZO DETECTADO ***",
                        (w // 2 - 210, h // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 3)
 
        # ── HUD ───────────────────────────────────────────────────────────────
        elapsed = int(time.time() - start_time)
        hud = [
            status_text,
            f"Frames ojos cerrados: {ear_counter}/{EAR_CONSEC_FRAMES}",
            f"Frames boca abierta:  {mar_counter}/{MAR_CONSEC_FRAMES}",
            f"Parpadeos: {total_blinks}   Bostezos: {total_bostezos}",
            f"Tiempo activo: {elapsed}s",
        ]
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (10, 25 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        status_color if i == 0 else (200, 200, 200), 2)
 
        # Barra EAR
        bar_w = int(np.clip(ear_avg / 0.5, 0, 1) * 180)
        cv2.rectangle(frame, (10, h - 50), (190, h - 34), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 50), (10 + bar_w, h - 34),
                      (0, 200, 0) if ear_avg >= EAR_THRESHOLD else (0, 0, 255), -1)
        cv2.putText(frame, "EAR", (195, h - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
 
        # Barra MAR
        bar_m = int(np.clip(mar_val / 1.0, 0, 1) * 180)
        cv2.rectangle(frame, (10, h - 28), (190, h - 12), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, h - 28), (10 + bar_m, h - 12),
                      (0, 200, 255) if mar_val >= MAR_THRESHOLD else (0, 180, 0), -1)
        cv2.putText(frame, "MAR", (195, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
 
        cv2.imshow("Detector de Somnolencia  [q = salir]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\nSesion terminada.")
    print(f"Parpadeos detectados: {total_blinks}")
    print(f"Bostezos detectados:  {total_bostezos}")
 
 
if __name__ == "__main__":
    run_detector()