#!/bin/bash


MODEL=projecte-aina/faster-whisper-large-v3-ca-3catparla # works fine
# MODEL=projecte-aina/faster-whisper-large-v3-tiny-caesar
# MODEL=large-v3
# MODEL=large-v2
PORT=8000
# MODEL_CASCADED_TRANSLATION=projecte-aina/aina-translator-ca-es
MODEL_CASCADED_TRANSLATION=projecte-aina/aina-translator-ca-en
LANGUAGE=ca # ca
BACKEND="faster-whisper" # "whisper-x" # faster-whisper, whisper-x

cmd="whisperlivekit-server --model $MODEL \
                           --port $PORT \
                           --model_cascaded_translation $MODEL_CASCADED_TRANSLATION  \
                           --language $LANGUAGE \
                           --backend $BACKEND \
                           --buffer_trimming_sec 10"
echo $cmd
eval $cmd