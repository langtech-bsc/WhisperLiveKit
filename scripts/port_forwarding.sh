#!/bin/bash

IP=84.88.51.151
PORT=8000
USER=marti

cmd="ssh -L $PORT:localhost:$PORT -N -o GatewayPorts=yes $USER@$IP"
echo $cmd
eval $cmd