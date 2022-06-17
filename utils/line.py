import os

import requests
from dotenv import load_dotenv


def main():
    send_line_notify("てすとてすと")


def send_line_notify(notification_message: str):
    """
    LINEに通知する
    """
    line_notify_token = get_token()
    line_notify_api = "https://notify-api.line.me/api/notify"
    headers = {"Authorization": f"Bearer {line_notify_token}"}
    data = {"message": f"message: {notification_message}"}
    res = requests.post(line_notify_api, headers=headers, data=data)
    assert res.status_code == 200


def get_token():
    load_dotenv()
    token = os.environ.get("LINE_TOKEN")
    assert token is not None
    return token


if __name__ == "__main__":
    main()
