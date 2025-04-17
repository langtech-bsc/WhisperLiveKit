#!/bin/bash


MODEL=projecte-aina/faster-whisper-large-v3-ca-3catparla
PORT=8000
MODEL_CASCADED_TRANSLATION=projecte-aina/aina-translator-ca-es
LANGUAGE=ca

cmd="whisperlivekit-server --model $MODEL \
                           --port $PORT \
                           --model_cascaded_translation $MODEL_CASCADED_TRANSLATION  \
                           --language $LANGUAGE"
echo $cmd
eval $cmd