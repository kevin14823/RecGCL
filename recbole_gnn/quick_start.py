import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color

from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer


from collections import defaultdict
import random
import torch
import numpy as np
import pandas as pd


def run_recbole_gnn(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """
    A fast running API, which includes the complete process of training and testing a model on a specified dataset.
    """
    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])

    num_noise = config['num_noise']
    noise = config['noise']

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    if noise:
        raw_user_list = dataset.inter_feat[dataset.uid_field].values
        raw_item_list = dataset.inter_feat[dataset.iid_field].values
        uid_field = dataset.uid_field
        iid_field = dataset.iid_field
        n_users = np.unique(dataset.inter_feat[uid_field].values).shape[0]
        n_items = np.unique(dataset.inter_feat[iid_field].values).shape[0]

        new_user_list, new_item_list, noise_pairs = add_noise_interactions(
            raw_user_list, raw_item_list, n_users, n_items, num_noise
        )

        if noise_pairs is None or len(noise_pairs) == 0:
            raise ValueError("ERROR: noise_pairs is empty after add_noise_interactions()!")

        logger.info(f"Generated noise pairs: {len(noise_pairs)}")

        new_inter_feat = pd.DataFrame({
            uid_field: new_user_list,
            iid_field: new_item_list
        })
        dataset.inter_feat = new_inter_feat

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    if noise:
        trainer.noise_pairs = noise_pairs

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation，將 noise_pairs 傳入 evaluate
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config['show_progress'], test=True)

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def add_noise_interactions(user_list, item_list, n_users, n_items, num_noise):
    user2items = defaultdict(list)
    for u, i in zip(user_list, item_list):
        user2items[u].append(i)

    new_user_list = []
    new_item_list = []
    noise_pairs = []

    for u, items in user2items.items():
        count = 0
        for i in items:
            new_user_list.append(u)
            new_item_list.append(i)
            count += 1
            if count % num_noise == 0:
                noise_item = random.randint(0, n_items - 1)
                while noise_item in items:
                    noise_item = random.randint(0, n_items - 1)
                new_user_list.append(u)
                new_item_list.append(noise_item)
                noise_pairs.append((u, noise_item))

    return new_user_list, new_item_list, noise_pairs

def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r""" The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'model': config['model'],
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
