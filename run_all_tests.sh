#! /usr/bin/bash

python3 evaluation.py -o results/ -m ff -i time-series --post-mapping --no-scaling
python3 evaluation.py -o results/ -m ff -i time-series --post-mapping --scaling
python3 evaluation.py -o results/ -m ff -i time-series --pre-mapping --no-scaling
python3 evaluation.py -o results/ -m ff -i time-series --pre-mapping --scaling

python3 evaluation.py -o results/ -m ff -i features --post-mapping --no-scaling
python3 evaluation.py -o results/ -m ff -i features --post-mapping --scaling
python3 evaluation.py -o results/ -m ff -i features --pre-mapping --no-scaling
python3 evaluation.py -o results/ -m ff -i features --pre-mapping --scaling

python3 evaluation.py -o results/ -m cnn -i time-series --post-mapping --no-scaling
python3 evaluation.py -o results/ -m cnn -i time-series --post-mapping --scaling
python3 evaluation.py -o results/ -m cnn -i time-series --pre-mapping --no-scaling
python3 evaluation.py -o results/ -m cnn -i time-series --pre-mapping --scaling
