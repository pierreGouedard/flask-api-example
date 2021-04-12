#!/usr/bin/python
# -*- coding: utf-8 -*-

# Global import
from flask import Flask, request, Response
from io import BytesIO, StringIO
import pandas as pd
import subprocess
import json
import yaml
import sys

# Local import

# Get settings
with open('conf/settings.yml') as f:
    settings = yaml.load(f, Loader=yaml.FullLoader)

api = Flask(__name__)


@api.route('/fit', methods=['POST'])
def fit():
    """
    Fit model and return tid for follow up

    :return: str
    """
    # Get path matching module
    path_fit = '/'.join([settings['project_dir'], settings['flaskapi_dir'], "fit_model.py"])

    # Read posted csv as DataFrame
    df = pd.read_csv(BytesIO(request.data), index_col=None)

    # millisecond timestamp
    tid = str(int(pd.Timestamp.now().timestamp() * 1000))

    # Launch match process and get tid
    proc = subprocess.Popen(
        ["python", path_fit, tid], stdin=subprocess.PIPE, stdout=sys.stdout, start_new_session=True
    )

    # Transmit df
    proc.stdin.write(df.to_csv().encode("utf-8"))
    proc.stdin.close()

    return tid


@api.route('/predict', methods=['POST'])
def predict():
    """
    predict submitted data
    :return: str
    """
    # Get id of model
    tid = request.args.get('tid', type=str)

    # Get path matching module
    path_predict = '/'.join([settings['project_dir'], settings['flaskapi_dir'], "predict.py"])

    # Read posted csv as DataFrame
    df = pd.read_csv(BytesIO(request.data), index_col=None)

    # Launch match process and get tid
    proc = subprocess.Popen(
        ["python", path_predict, tid], stdin=subprocess.PIPE, stdout=sys.stdout, start_new_session=True
    )

    proc.stdin.write(df.to_csv().encode("utf-8"))
    proc.stdin.close()

    return tid


@api.route('/track', methods=['GET'])
def track():
    """
    Return fitting info from args tid.

    :return: text/json info
    """
    # Get path matching meta info
    tid = request.args.get('tid', type=str)
    path_track = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['meta'].format(tid)])

    try:
        with open(path_track, 'r') as handle:
            d_info = json.load(handle)
    except IOError:
        return Response(
            json.dumps({'stderr': 'No meta info for matching with tid {}'.format(tid)}), mimetype='text/json'
        )

    return Response(json.dumps(d_info), mimetype='text/json')


@api.route('/download', methods=['GET'])
def download():
    """
    Return prediction

    :return: text/csv
    """
    tid = request.args.get('tid', type=str)
    path_download = '/'.join([settings['project_dir'], settings['data_dir'], settings['outfiles']['pred'].format(tid)])
    try:
        df = pd.read_csv(path_download)
        with StringIO() as f:
            df.to_csv(f)
            return Response(f.getvalue(), mimetype='text/csv')

    except FileNotFoundError:
        return Response("No matching for tid {} to download".format(tid), mimetype='text/csv')
