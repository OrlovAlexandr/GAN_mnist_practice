import threading
from pathlib import Path

from config import Optimizer
from config import Strategy
from config import cfg
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request
from main import start_train
from utils.parameters import stop_training_flag
from utils.parameters import update_config


app = Flask(__name__)
train_thread = None


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', cfg=cfg, Optimizer=Optimizer, Strategy=Strategy)


@app.route('/update_config', methods=['POST'])
def update_config_route():
    data = request.json
    update_config(data)
    return jsonify(success=True)


@app.route('/start_training', methods=['POST'])
def start_training():
    global train_thread  # noqa: PLW0603
    if train_thread is None or not train_thread.is_alive():
        train_thread = threading.Thread(target=start_train)
        train_thread.start()
    return jsonify(success=True)


@app.route('/stop_training', methods=['POST'])
def stop_training():
    global train_thread  # noqa: PLW0602
    stop_training_flag.set()
    if train_thread and train_thread.is_alive():
        train_thread.join()
    return jsonify(success=True)


@app.route('/get_images', methods=['GET'])
def get_images():
    images_dir = Path() / 'static' / 'images' / 'trains' / f'train_{cfg.now}' / 'images'
    images = sorted(images_dir.glob('*.png'), key=lambda x: x.stat().st_mtime)
    if images:
        latest_image = str(images[-1])  # Получить путь к последнему изображению
        return jsonify(image=latest_image)
    return jsonify(image=None)


if __name__ == '__main__':
    app.run(debug=True)  # noqa: S201
