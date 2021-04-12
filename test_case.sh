#!/usr/bin/env bash
set -e

EXT_DATA='.csv'
EXT_INFO='.json'

function log {
    local PURPLE='\033[0;35m'
    local NOCOLOR='\033[m'
    local BOLD='\033[1m'
    local NOBOLD='\033[0m'
    echo -e -n "${PURPLE}${BOLD}$1${NOBOLD}${NOCOLOR}"
}

function fit {
    log "POST: fitting $SUBMISSION at localhost:5000/fit... \\n\\n"
    TID=$(curl \
        --silent \
        --fail \
        --data-binary @$SUBMISSION \
        --header "Content-Type: text/csv" \
        "localhost:5000/fit")

    log "Fitting initiated, you can track process using track command and tid: $TID\\n"
}

function predict {
    log "POST: predicting $SUBMISSION at localhost:5000/predict... \\n\\n"
    curl \
        --silent \
        --fail \
        --data-binary @$SUBMISSION \
        --header "Content-Type: text/csv" \
        "localhost:5000/predict?tid=$TID"

    log "Prediction initiated, you can download preds using download command and tid: $TID\\n"
}

function track_fitting {
    log "\\ GET: track fitting info $TID to localhost:5000/track > <tid>.json... \\n\\n"
    curl \
        --silent \
        --fail \
        "localhost:5000/track?tid=$TID"\
        > "$OUT"
    log "Info fitting downloaded in $OUT \\n"
}

function download_prediction {
    log "\\ GET: Download prediction $TID to localhost:5000/download > <tid>.csv... \\n\\n"
    curl \
        --silent \
        --fail \
        "localhost:5000/download?tid=$TID" \
        > "$OUT"
    log "Prediction downloaded at $OUT \\n"
}

case $1 in

  fit)
    SUBMISSION=$2
    fit || true
    ;;

  predict)
    SUBMISSION=$2
    TID=$3
    predict || true
    ;;

  track | download)
    TID=$2

    if [ "$1" == track ]
    then
        OUT="$TID$EXT_INFO"
        track_fitting
    else
        OUT="$TID$EXT_DATA"
        download_prediction
    fi
    ;;

  *)
    echo "Choose second arguments among ['fit', 'predict', 'track', 'download']"
    ;;
esac