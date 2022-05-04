import argparse
import sys
import matplotlib
import os
import os.path as osp


import architecture.train as train

"""
Usage example (training):
    ~ Create new directory for output storage, for example 'model-1_output'.
    ~ Run in command line:
        python3 train.py \
            --trainer_name=Trainer_Debug_V1 \
            --output_dir=model-1_output \
            --train_csv=2022-mar-31_data-debug_train.csv \
            --val_csv=2022-mar-31_data-debug_val.csv \
            --dataset_root=/workspaces/Dataset \
            --device=cpu \
            --run_mode=d
"""


parser = argparse.ArgumentParser()
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
parser.add_argument('--device', required=False, default=['cpu'], nargs='+',
    help='Device(s) for running torch. Either cpu or cuda:n where n is '
         'the gpu\'s id or list of such devices.')
parser.add_argument('--run_mode', required=False, default='p', 
    choices=['p', 'd'], help='Run mode, either production or debug.')

def main(args):
    # Setup backend in order to work inside Docker without having available display.
    matplotlib.use('Agg')

    args = parser.parse_args(args)
    assert isinstance(args.output_dir, str) and osp.isdir(args.output_dir),\
        f'Please specify existing directory with --output_dir.' \
        f' {args.output_dir} does not exist.'
    assert isinstance(args.trainer_name, str) and \
        args.trainer_name in train.__dict__
    trainer_class = getattr(train, args.trainer_name)
    trainer = trainer_class()
    trainer.output_dir = args.output_dir
    trainer.train_csv = args.train_csv
    trainer.val_csv = args.val_csv
    trainer.dataset_root = args.dataset_root
    trainer.device = args.device
    trainer.RUN_MODE = args.run_mode
    
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])