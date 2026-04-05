import os
import requests

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    r = requests.post(url, data=data, timeout=10)
    print(r.status_code)
    print(r.text)

if __name__ == "__main__":
    send_telegram("✅ GitHub Actions 텔레그램 테스트 성공")
