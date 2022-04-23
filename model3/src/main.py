import argparse
import sys
import matplotlib
import os
import os.path as osp

import architecture.train as train
#from architecture.inference.inference import evaluate_model

"""
Usage example (training):
    ~ Create new directory for output storage, for example 'model-1_output'.
    ~ Run in command line:
        python3 main.py \
            --action=train \
            --output_dir=model-1_output \
            --train_csv=2022-mar-31_data_train.csv \
            --val_csv=2022-mar-31_data_val.csv

Usage example (inference):
    ~ TODO
"""


parser = argparse.ArgumentParser()
parser.add_argument('--action', required=True, choices=['train', 'inference'],
    help="Action to be performed.")
parser.add_argument('--trainer_name', required=False,
    help='Class name of the trainer which will perform the training.')
parser.add_argument('--train_csv', required=False, 
    help='Filename of the training csv.')
parser.add_argument('--val_csv', required=False,
    help='Filename of the validation csv.')
parser.add_argument('--output_dir', required=False,
    help='Path to existing directory in which all the output will be saved.')
parser.add_argument('--parameters_path', required=False,
    help='Path to model parameters.')
parser.add_argument('--dataset_root', required=False, 
    help='Root directory of the dataset.')

def main(args):
    # Setup backend in order to work inside Docker without having available display.
    matplotlib.use('Agg')

    args = parser.parse_args(args)
    if(args.action == 'train'):
        assert isinstance(args.output_dir, str) and osp.isdir(args.output_dir),\
            'Please specify existing directory with --output_dir.'
        assert isinstance(args.trainer_name, str) and \
            args.trainer_name in train.__dict__
        trainer_class = getattr(train, args.trainer_name)
        trainer = trainer_class()
        trainer.output_dir = args.output_dir
        trainer.train_csv = args.train_csv
        trainer.val_csv = args.val_csv
        trainer.dataset_root = args.dataset_root
        trainer.train()

    else:
        raise NotImplementedError
        assert isinstance(args.output_dir, str) and osp.isdir(args.output_dir),\
            'Please specify existing directory with --output_dir.'
        evaluate_model(args.parameters_path, args.output_dir, args.val_csv)
    

if __name__ == "__main__":
    main(sys.argv[1:])