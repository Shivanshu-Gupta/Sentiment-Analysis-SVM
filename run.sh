#!/bin/bash

INPUT_FILE=$1
INPUT_FILE_CLEANED=$INPUT_FILE.clean
INPUT_FILE_PREPROCESSED=$INPUT_FILE.preprocessed

./clean.sh $INPUT_FILE

python preprocess.py --input_file $INPUT_FILE_CLEANED --output_file $INPUT_FILE_PREPROCESSED
python test.py --test_data $INPUT_FILE_PREPROCESSED --load_dir all_data/ --output_file $2
