import requests
import json

# ✨ 請改成你機器人的 IP
KEBBI_IP = "172.20.10.8"

def move(x, y, theta):
    url = f"http://{KEBBI_IP}:10000/v1/robot/move"
    data = {"x": x, "y": y, "theta": theta}
    res = requests.post(url, json=data)
    print("[移動] 回應：", res.text)

def speak(text):
    url = f"http://{KEBBI_IP}:10000/v1/robot/speak"
    data = {"text": text}
    res = requests.post(url, json=data)
    print("[說話] 回應：", res.text)

def face(expression):
    url = f"http://{KEBBI_IP}:10000/v1/robot/face"
    data = {"name": expression}
    res = requests.post(url, json=data)
    print("[表情] 回應：", res.text)

def motion(name):
    url = f"http://{KEBBI_IP}:10000/v1/robot/motion/play"
    data = {"name": name}
    res = requests.post(url, json=data)
    print("[動作] 回應：", res.text)

# === 測試 ===
move(300, 0, 0)               # 向前走 30 公分
speak("你好，我是酷比機器人")    # 開口說話
face("happy")                # 表情顯示高興
motion("wave")               # 揮手（動作名根據內建動作庫）
