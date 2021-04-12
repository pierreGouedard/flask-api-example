#!/usr/bin/python
# -*- coding: utf-8 -*-

# Global import
import sys
import pandas as pd
import pickle
import yaml
from io import StringIO

# Local import
with open('conf/settings.yml') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)


def predict(df, tid):
    """

    :param df:
    :param tid:
    :return:
    """
    # Load model
    model_path = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['model'].format(tid)])
    pred_path = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['pred'].format(tid)])

    try:
        with open(model_path, 'rb') as handle:
            clf = pickle.load(handle)
    except FileNotFoundError:
        sys.stdout.write("No model found for {}".format(tid))
        return

    # Predict
    df_pred = clf.predict(df)

    # Save prediction
    df_pred.to_csv(pred_path, index=True)


if __name__ == "__main__":
    # Get csv string
    str_df = sys.stdin.read()
    sys.stdin.close()

    # Build Dataframe and launch matching sub process
    df = pd.read_csv(StringIO(str_df), encoding='utf8')

    df = pd.read_csv('data/iris.csv')
    predict(df, int(sys.argv[1]))
