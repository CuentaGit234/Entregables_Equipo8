import smtplib
from email.mime.text import MIMEText

GMAIL_ORIGEN   = "bigpatch09@gmail.com"
GMAIL_PASSWORD = "erad ppip opjy rehb"
CORREO_DESTINO = "bigpatch09@gmail.com"

print("Iniciando prueba...")

msg = MIMEText("Prueba del detector de somnolencia")
msg["Subject"] = "Prueba alerta somnolencia"
msg["From"]    = GMAIL_ORIGEN
msg["To"]      = CORREO_DESTINO

print("Conectando a Gmail...")

try:
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as s:
        print("Conexion exitosa, haciendo login...")
        s.login(GMAIL_ORIGEN, GMAIL_PASSWORD)
        print("Login exitoso, enviando...")
        s.sendmail(GMAIL_ORIGEN, CORREO_DESTINO, msg.as_string())
        print("Correo enviado OK")
except smtplib.SMTPAuthenticationError:
    print("Error: contrasena de aplicacion incorrecta")
except smtplib.SMTPConnectError:
    print("Error: no se puede conectar a Gmail, revisa el internet")
except Exception as e:
    print(f"Error inesperado: {type(e).__name__}: {e}")

print("Fin del script")