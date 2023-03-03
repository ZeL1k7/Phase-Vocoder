#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 input.wav output.wav time_stretch_ratio"
  exit 1
fi

input_file=$1
output_file=$2
ratio=$3

pip install -r requirements.txt
python vocoder.py $input_file $output_file $ratio
