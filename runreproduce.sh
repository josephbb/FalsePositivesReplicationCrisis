#!/bin/sh

if [ "$1" = "run-reproduce" ]; then
  cd ./replicationsurveys
  exec python ./reproduce.py
elif [ "$1" = "jupyter" ]; then
  shift  
  exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root "$@"
else
  exec "$@"
fi