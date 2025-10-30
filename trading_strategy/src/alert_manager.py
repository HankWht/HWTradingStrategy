import smtplib
from email.mime.text import MIMEText
from telegram import Bot
import os

class AlertManager:
    def __init__(self, email_conf=None, telegram_token=None, telegram_chat_id=None):
        self.email_conf = email_conf
        self.telegram_bot = Bot(token=telegram_token) if telegram_token else None
        self.chat_id = telegram_chat_id

    def send_email(self, subject, message):
        if not self.email_conf:
            print("⚠️ Configuración de correo no definida.")
            return

        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = self.email_conf["sender"]
        msg["To"] = self.email_conf["recipient"]

        with smtplib.SMTP(self.email_conf["server"], self.email_conf["port"]) as server:
            server.starttls()
            server.login(self.email_conf["sender"], self.email_conf["password"])
            server.send_message(msg)

        print("📨 Alerta enviada por correo.")

    def send_telegram(self, message):
        if self.telegram_bot and self.chat_id:
            self.telegram_bot.send_message(chat_id=self.chat_id, text=message)
            print("📲 Alerta enviada por Telegram.")
