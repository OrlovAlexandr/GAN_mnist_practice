import threading
from pathlib import Path

import pandas as pd
from config import Optimizer
from config import SequenceType
from config import Strategy
from config import cfg
from config import get_time


def create_params_dict() -> dict:
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}


def create_train_path() -> Path:
    train_path = Path() / 'static' / 'trains' / f'train_{cfg.now}'
    train_path.mkdir(parents=True, exist_ok=True)
    params_dict = create_params_dict()
    params_dict['time'] = cfg.now
    df_params = pd.DataFrame(params_dict, index=['params']).transpose()
    df_params.to_csv(train_path / 'parameters.csv')
    return train_path


def update_config(data):
    cfg.learn_rate = float(data.get('learn_rate', cfg.learn_rate))
    cfg.noise_size = int(data.get('noise_size', cfg.noise_size))
    cfg.conditional = str(data.get('conditional', 'false')).lower() == 'true'
    cfg.batch_size = int(data.get('batch_size', cfg.batch_size))
    cfg.num_epochs = int(data.get('num_epochs', cfg.num_epochs))
    cfg.optimizer = Optimizer[data.get('optimizer', cfg.optimizer.name)]
    cfg.strategy = Strategy[data.get('strategy', cfg.strategy.name)]
    cfg.sequence_type = SequenceType[data.get('sequence_type', cfg.sequence_type.name)]
    if cfg.optimizer == Optimizer.ADAM:
        cfg.adam_beta1 = float(data.get('adam_beta1', cfg.adam_beta1))
        cfg.adam_beta2 = float(data.get('adam_beta2', cfg.adam_beta2))
    if cfg.strategy == Strategy.CLIP_WEIGHT:
        cfg.clip_value = float(data.get('clip_value', cfg.clip_value))
    if cfg.strategy == Strategy.GRAD_PENALTY:
        cfg.gp_lambda = int(data.get('gp_lambda', cfg.gp_lambda))
    cfg.now = get_time()


stop_training_flag = threading.Event()


def stop_training():
    stop_training_flag.set()
