#!/bin/bash

# MODEL=projecte-aina/faster-whisper-large-v3-ca-3catparla
# MODEL=projecte-aina/faster-whisper-large-v3-tiny-caesar # works fine
# MODEL=large-v3
# MODEL=large-v2
MODEL=BSC-LT/faster-whisper-large-v3-ca-punctuated-3370h
PORT=8000
# MODEL_CASCADED_TRANSLATION=projecte-aina/aina-translator-ca-es
MODEL_CASCADED_TRANSLATION=projecte-aina/aina-translator-ca-en
LANGUAGE=ca # ca
BACKEND="faster-whisper" # "whisper-x" # faster-whisper, whisper-x
SIMULATION_FILE="/home/marti/projects/langtech-bsc/WhisperLiveKit/rXn1vkhUZZk_chunk_300s.wav"

cmd="whisperlivekit-server --model $MODEL \
                           --port $PORT \
                           --model_cascaded_translation $MODEL_CASCADED_TRANSLATION  \
                           --language $LANGUAGE \
                           --backend $BACKEND \
                           --buffer_trimming_sec 10 " # \
                           #--simulation_file $SIMULATION_FILE"
echo $cmd
eval $cmd