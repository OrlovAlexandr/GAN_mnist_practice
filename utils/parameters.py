from pathlib import Path

import pandas as pd
from config import cfg


def create_params_dict() -> dict:
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith('_')}


def create_train_path() -> Path:
    train_path = Path() / 'imgs' / 'trains' / f'train_{cfg.now}'
    train_path.mkdir(parents=True, exist_ok=True)
    params_dict = create_params_dict()
    params_dict['time'] = cfg.now
    df_params = pd.DataFrame(params_dict, index=['params']).transpose()
    df_params.to_csv(train_path / 'parameters.csv')
    return train_path
