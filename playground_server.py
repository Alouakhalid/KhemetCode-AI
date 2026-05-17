from flask import Flask, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def playground():
    return send_from_directory("static", "playground.html")

if __name__ == "__main__":
    print("🔺 KhemetCode Playground Server starting on http://127.0.0.1:5051")
    app.run(host="0.0.0.0", port=5051, debug=False)
