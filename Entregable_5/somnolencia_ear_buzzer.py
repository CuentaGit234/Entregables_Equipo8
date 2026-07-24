# ============================================================
# Detector de somnolencia por EAR + tiempo de ojos cerrados
# Compatible con Raspberry Pi 3 + Python + Thonny
# Activa buzzer de 5V mediante GPIO cuando detecta somnolencia
# Autor: Dranzer
# ============================================================

import cv2
import dlib
import time
import math
import os
from collections import deque

# ============================================================
# CONFIGURACION PRINCIPAL
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

FLIP_IMAGE = True

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Umbral EAR.
# Si detecta ojos cerrados cuando estan abiertos, bajar a 0.19 o 0.20.
# Si no detecta bien los ojos cerrados, subir a 0.22 o 0.23.
EAR_THRESHOLD = 0.21

# Tiempo minimo con ojos cerrados para activar somnolencia
CLOSED_EYES_SECONDS = 2.0

# Suavizado del EAR para reducir ruido
SMOOTHING_FRAMES = 3

SHOW_EYE_POINTS = True

TRY_MULTIPLE_CAMERAS = True

# ============================================================
# CONFIGURACION DEL BUZZER 5V
# ============================================================

# Activar salida fisica
USE_BUZZER = True

# GPIO BCM 18 = pin fisico 12 de la Raspberry Pi
BUZZER_PIN = 18

# True: salida HIGH activa el buzzer.
# Usar True si manejas el buzzer con transistor NPN.
# Usar False si tienes un modulo buzzer activo en LOW.
BUZZER_ACTIVE_HIGH = True

# True: buzzer intermitente
# False: buzzer continuo mientras haya somnolencia
BUZZER_BEEP_MODE = True

# Tiempos del pitido intermitente
BUZZER_ON_TIME = 0.25
BUZZER_OFF_TIME = 0.25

GPIO_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(eye_points):
    """
    Calcula el EAR de un ojo usando 6 puntos.

    EAR = (dist(p2,p6) + dist(p3,p5)) / (2 * dist(p1,p4))
    """

    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def shape_to_points(shape, indexes):
    points = []

    for i in indexes:
        x = shape.part(i).x
        y = shape.part(i).y
        points.append((x, y))

    return points


def draw_eye(frame, eye_points):
    for point in eye_points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    for i in range(len(eye_points)):
        p1 = eye_points[i]
        p2 = eye_points[(i + 1) % len(eye_points)]
        cv2.line(frame, p1, p2, (0, 255, 0), 1)


def put_text(frame, text, x, y, color=(255, 255, 255), scale=0.55, thickness=1):
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness
    )


def find_haar_cascade():
    possible_paths = []

    try:
        if hasattr(cv2, "data"):
            possible_paths.append(
                os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            )
    except Exception:
        pass

    possible_paths.append("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    possible_paths.append("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    possible_paths.append("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
    possible_paths.append("/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    possible_paths.append("haarcascade_frontalface_default.xml")

    for path in possible_paths:
        if path is not None and os.path.exists(path):
            return path

    return None


def setup_buzzer():
    if not USE_BUZZER:
        return

    if not GPIO_AVAILABLE:
        print("GPIO no disponible. El programa seguira sin buzzer.")
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)

    set_buzzer_output(False)


def set_buzzer_output(active):
    if not USE_BUZZER or not GPIO_AVAILABLE:
        return

    if BUZZER_ACTIVE_HIGH:
        GPIO.output(BUZZER_PIN, GPIO.HIGH if active else GPIO.LOW)
    else:
        GPIO.output(BUZZER_PIN, GPIO.LOW if active else GPIO.HIGH)


def update_buzzer(alarm_active, current_time):
    """
    Controla el buzzer.
    Si BUZZER_BEEP_MODE es True, hace pitidos intermitentes.
    Si es False, el buzzer queda continuo mientras alarm_active sea True.
    """

    if not USE_BUZZER or not GPIO_AVAILABLE:
        return

    if not alarm_active:
        set_buzzer_output(False)
        return

    if not BUZZER_BEEP_MODE:
        set_buzzer_output(True)
        return

    cycle_time = BUZZER_ON_TIME + BUZZER_OFF_TIME
    position = current_time % cycle_time

    if position < BUZZER_ON_TIME:
        set_buzzer_output(True)
    else:
        set_buzzer_output(False)


def cleanup_buzzer():
    if not USE_BUZZER or not GPIO_AVAILABLE:
        return

    set_buzzer_output(False)
    GPIO.cleanup()


def get_largest_face(faces):
    if len(faces) == 0:
        return None

    largest_face = None
    largest_area = 0

    for face in faces:
        x, y, w, h = face
        area = w * h

        if area > largest_area:
            largest_area = area
            largest_face = face

    return largest_face


def open_camera():
    """
    Abre la camara usando V4L2 para evitar errores de GStreamer.
    Prueba CAMERA_INDEX y, opcionalmente, otros indices.
    """

    if TRY_MULTIPLE_CAMERAS:
        camera_indexes = [CAMERA_INDEX, 0, 1, 2, 3]
    else:
        camera_indexes = [CAMERA_INDEX]

    used_indexes = []

    for index in camera_indexes:
        if index in used_indexes:
            continue

        used_indexes.append(index)

        print("Probando camara index:", index)

        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

        if not cap.isOpened():
            print("No se pudo abrir /dev/video{}".format(index))
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        try:
            fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        time.sleep(1.0)

        ok_frame = False

        for _ in range(10):
            ret, frame = cap.read()

            if ret and frame is not None:
                ok_frame = True
                break

            time.sleep(0.1)

        if ok_frame:
            print("Camara funcionando en index:", index)
            return cap

        print("La camara abrio, pero no entrega frames en index:", index)
        cap.release()

    return None


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not os.path.exists(PREDICTOR_PATH):
        print("ERROR: No se encontro el archivo:")
        print(PREDICTOR_PATH)
        print("")
        print("Coloca shape_predictor_68_face_landmarks.dat en la misma carpeta del programa.")
        return

    print("Cargando modelo de landmarks...")
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    print("Buscando Haar Cascade de OpenCV...")
    cascade_path = find_haar_cascade()

    if cascade_path is None:
        print("ERROR: No se encontro haarcascade_frontalface_default.xml")
        print("")
        print("Ejecuta:")
        print("sudo apt install -y opencv-data")
        return

    print("Haar Cascade encontrado en:")
    print(cascade_path)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("ERROR: No se pudo cargar Haar Cascade de OpenCV.")
        return

    print("Abriendo camara...")
    cap = open_camera()

    if cap is None:
        print("ERROR: No se pudo obtener video de ninguna camara.")
        print("")
        print("Verifica con estos comandos:")
        print("ls /dev/video*")
        print("v4l2-ctl --list-devices")
        print("fswebcam -d /dev/video0 -r 320x240 prueba.jpg")
        return

    setup_buzzer()

    RIGHT_EYE_INDEXES = [36, 37, 38, 39, 40, 41]
    LEFT_EYE_INDEXES = [42, 43, 44, 45, 46, 47]

    ear_buffer = deque(maxlen=SMOOTHING_FRAMES)

    closed_start_time = None
    closed_elapsed = 0.0
    alarm_active = False

    frame_counter = 0
    fps = 0.0
    fps_start_time = time.time()

    print("Programa iniciado.")
    print("Presiona la tecla 'q' en la ventana de video para salir.")

    try:
        while True:
            ret, frame = cap.read()
            current_time = time.time()

            if not ret or frame is None:
                print("No se pudo leer frame de la camara.")
                update_buzzer(False, current_time)
                time.sleep(0.1)
                continue

            if FLIP_IMAGE:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            largest_face = get_largest_face(faces)

            if largest_face is not None:
                x, y, w, h = largest_face

                dlib_rect = dlib.rectangle(
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                )

                shape = predictor(gray, dlib_rect)

                right_eye = shape_to_points(shape, RIGHT_EYE_INDEXES)
                left_eye = shape_to_points(shape, LEFT_EYE_INDEXES)

                right_ear = calculate_ear(right_eye)
                left_ear = calculate_ear(left_eye)

                ear = (right_ear + left_ear) / 2.0

                ear_buffer.append(ear)
                smooth_ear = sum(ear_buffer) / len(ear_buffer)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

                if SHOW_EYE_POINTS:
                    draw_eye(frame, right_eye)
                    draw_eye(frame, left_eye)

                if smooth_ear < EAR_THRESHOLD:
                    if closed_start_time is None:
                        closed_start_time = current_time

                    closed_elapsed = current_time - closed_start_time

                    if closed_elapsed >= CLOSED_EYES_SECONDS:
                        alarm_active = True
                    else:
                        alarm_active = False

                else:
                    closed_start_time = None
                    closed_elapsed = 0.0
                    alarm_active = False

                update_buzzer(alarm_active, current_time)

                put_text(frame, "EAR: {:.3f}".format(smooth_ear), 10, 25)

                put_text(
                    frame,
                    "Tiempo ojos cerrados: {:.1f}s".format(closed_elapsed),
                    10,
                    50
                )

                if smooth_ear < EAR_THRESHOLD:
                    put_text(frame, "OJOS CERRADOS", 10, 75, color=(0, 255, 255))
                else:
                    put_text(frame, "OJOS ABIERTOS", 10, 75, color=(0, 255, 0))

                if alarm_active:
                    put_text(
                        frame,
                        "ALERTA: SOMNOLENCIA DETECTADA",
                        10,
                        110,
                        color=(0, 0, 255),
                        scale=0.65,
                        thickness=2
                    )

                    put_text(
                        frame,
                        "BUZZER ACTIVADO",
                        10,
                        140,
                        color=(0, 0, 255),
                        scale=0.65,
                        thickness=2
                    )

                    cv2.rectangle(
                        frame,
                        (2, 2),
                        (frame.shape[1] - 2, frame.shape[0] - 2),
                        (0, 0, 255),
                        3
                    )

            else:
                ear_buffer.clear()
                closed_start_time = None
                closed_elapsed = 0.0
                alarm_active = False
                update_buzzer(False, current_time)

                put_text(frame, "Rostro no detectado", 10, 25, color=(0, 0, 255))
                put_text(frame, "Ajusta camara o iluminacion", 10, 50, color=(0, 0, 255))

            frame_counter += 1
            elapsed_fps_time = time.time() - fps_start_time

            if elapsed_fps_time >= 1.0:
                fps = frame_counter / elapsed_fps_time
                frame_counter = 0
                fps_start_time = time.time()

            put_text(frame, "FPS: {:.1f}".format(fps), 10, frame.shape[0] - 10)

            cv2.imshow("Detector de Somnolencia - EAR + Buzzer", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("Programa detenido por teclado.")

    finally:
        update_buzzer(False, time.time())
        cleanup_buzzer()
        cap.release()
        cv2.destroyAllWindows()
        print("Programa finalizado.")


if __name__ == "__main__":
    main()