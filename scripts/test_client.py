import requests
import base64
import json
import sys
import os

# 配置
SERVER_URL = "http://localhost:8080/ocr"
IMAGE_PATH = "test.jpg"  # 替换为您的测试图像（根目录）

def test_ocr():
    if not os.path.exists(IMAGE_PATH):
        print(f"图像 {IMAGE_PATH} 不存在！")
        return

    with open(IMAGE_PATH, "rb") as f:
        img_data = f.read()
    base64_img = base64.b64encode(img_data).decode('utf-8')

    payload = {"image_base64": base64_img}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(SERVER_URL, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            result = response.json()
            print("OCR 结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"错误 {response.status_code}: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_ocr()