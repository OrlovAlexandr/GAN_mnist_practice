import threading
from pathlib import Path

from config import cfg
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from flask import send_from_directory
from main import start_train


app = Flask(__name__)

training_started = False
image_path = Path() / 'static' / 'images' / 'trains' / f'train_{cfg.now}' / 'images'


@app.route("/")
def index():
    return render_template("index.html", config=cfg)


@app.route('/start_training', methods=['POST'])
def start_training():
    global training_started  # noqa: PLW0603
    if not training_started:
        training_thread = threading.Thread(target=start_train)
        training_thread.start()
        training_started = True
        return jsonify({"status": "success", "message": "Training started!"})
    return jsonify({"status": "error", "message": "Training already in progress."})


@app.route("/update_config", methods=["POST"])
def update_config():
    new_config = request.form.to_dict()
    for key, value in new_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, type(getattr(cfg, key))(value))
    return jsonify({"status": "success", "message": "Configuration updated successfully"})


@app.route("/get_latest_image")
def get_latest_image():
    if not image_path.exists():
        return jsonify({"image_url": ""})
    # check if there are any images
    if not any(image_path.iterdir()):
        return jsonify({"image_url": ""})
    latest_image = sorted(image_path.glob('*.png'))[-1]
    return jsonify({"image_url": str(latest_image)})


@app.route(f'/{image_path!s}/<path:filename>')
def static_images(filename):
    return send_from_directory(str(image_path), filename)


if __name__ == "__main__":
    app.run(debug=False)
