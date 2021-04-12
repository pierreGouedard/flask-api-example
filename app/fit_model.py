#!/usr/bin/python
# -*- coding: utf-8 -*-

# Global import
import json
import sys
import pandas as pd
import yaml
from ml_ops.clf_model import ClfSelector
from io import StringIO

# Local import
with open('conf/settings.yml') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)


def fit(df: pd.DataFrame, tid: str) -> None:
    """
    Fit classifier

    :param df: DataFrame, training data
    :param tid: str, id of the model

    :return: None
    """
    # Get paths
    meta_path = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['meta'].format(tid)])
    model_path = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['model'].format(tid)])
    update_info(meta_path, {})

    # complete transform params
    param_transform = settings["param_transform"]
    param_transform['num_cols'] = settings['num_cols']
    param_transform['cat_cols'] = settings['cat_cols']
    param_transform['target_col'] = settings['target_col']
    param_transform['labels'] = list(df[settings['target_col']].loc[~df[settings['target_col']].isna()].unique())

    # Instantiate selection model
    clfs = ClfSelector(
        df_data=df,
        param_mdl=settings['model_param'],
        param_mdl_grid=settings['model_param_grid'],
        param_transform=param_transform,
        param_transform_grid=settings['param_transform_grid'],
        params_fold=settings['param_fold'],
        scoring=settings['scoring']
    )

    for meta in clfs.fit():
        if type(meta) != bool:
            update_info(meta_path, meta)

    # Save classifer
    clfs.save_classifier(model_path)


def update_info(path: str, meta: dict) -> None:
    """
    Dump information in json format.

    :param path: str
    :param meta: dict, train info

    :return: None
    """
    with open(path, 'w') as handle:
        json.dump(meta, handle)


if __name__ == "__main__":
    # Get csv string
    str_df = sys.stdin.read()
    sys.stdin.close()

    # Build Dataframe and launch matching sub process
    df = pd.read_csv(StringIO(str_df), encoding='utf8')

    fit(df, str(sys.argv[1]))
