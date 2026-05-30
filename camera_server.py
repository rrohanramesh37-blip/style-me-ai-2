"""
camera_server.py — Tiny Flask server that serves the camera capture page
and accepts the captured JPEG, saving it to captured_face.jpg
Run alongside the Streamlit app.
"""
from flask import Flask, request, jsonify, send_from_directory
import base64, os, re

app = Flask(__name__, static_folder=".")
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captured_face.jpg")

@app.route("/")
def index():
    return send_from_directory(".", "camera.html")

@app.route("/capture", methods=["POST"])
def capture():
    data = request.get_json()
    img_data = data.get("image", "")
    # Strip data URL header  e.g. "data:image/jpeg;base64,..."
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]
    with open(SAVE_PATH, "wb") as f:
        f.write(base64.b64decode(img_data))
    return jsonify({"status": "ok", "path": SAVE_PATH})

@app.route("/status")
def status():
    exists = os.path.exists(SAVE_PATH)
    mtime = os.path.getmtime(SAVE_PATH) if exists else 0
    return jsonify({"captured": exists, "mtime": mtime})

@app.route("/clear", methods=["POST"])
def clear():
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    print("Camera server running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
