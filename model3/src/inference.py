import architecture.train as train
import argparse
import sys
import os
import os.path as osp
import pandas as pd
import torch
from architecture.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib

"""
Usage example (inference):
    ~ Create a new directory for output storage, for example 'model-1_output'.
    ~ Run in command line:
        python3 inference.py
            --trainer_name=Trainer_Debug_V1 \
            --parameters_path=outputs/debug_model_parameters.prms \
            --dataset_root=/workspaces/Dataset \
            --target_csv=2022-may-04_minibatch_n8_inference.csv \
            --output_dir=model-1_output \
            --device=cpu \
            --batch_size=1 \
            --num_workers=4
"""

parser = argparse.ArgumentParser()
parser.add_argument('--trainer_name', required=True,
    help='Class name of the trainer which was used for the training.')
parser.add_argument('--parameters_path', required=True,
    help='Path to model parameters.')    

parser.add_argument('--dataset_root', required=False, 
    help='Root directory of the dataset.')
parser.add_argument('--target_csv', required=False,
    help='Filename of either the validation csv (columns uuid, '
    'latitude, longitude) or just csv with list of images (column uuid).')

parser.add_argument('--output_dir', required=False,
    help='Path to existing directory in which all the output will be saved.')
parser.add_argument('--device', required=False, default=['cpu'], nargs='+',
    help='Device(s) for running torch. Either cpu or cuda:n where n is '
         'the gpu\'s id or list of such devices.')
parser.add_argument('--batch_size', required=False, default=1, 
    help='Batch size.')
parser.add_argument('--num_workers', required=False, default=1, 
    help='Number of workers.')

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
    device = [torch.device(dev) for dev in args.device]
    dataset = Dataset(
        args.dataset_root, 
        csv_file=args.target_csv,
        device=device,
        **trainer.dataset_val_kwds)
    
    model = trainer.model
    output = inference(
        dataset=dataset,
        model=model,
        parameters_path=args.parameters_path,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device)
    
    if dataset.contains_gt:
        output['gt_latitude'] = dataset.data['latitude']
        output['gt_longitude'] = dataset.data['longitude']
        output = output[[
            'uuid', 'gt_latitude', 'gt_longitude', 
            'mo_latitude', 'mo_longitude']]
    
    output.to_csv(osp.join(args.output_dir, 'output.csv'), index=False)
    
def inference(dataset, model, parameters_path, batch_size, num_workers, device):
    if len(device) > 1:
        raise NotImplementedError("Multi GPU not yet supported")
    model.load_state_dict(torch.load(parameters_path, map_location=device[0]))
    model.to(device[0])

    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False,
        num_workers=num_workers, shuffle=False)
    
    model.eval()
    result_df = pd.DataFrame(
            columns=['uuid', 'mo_latitude', 'mo_longitude'])
    for batch_ind, batch in enumerate(tqdm(dataloader)):
        input = {
            'image_N': batch['image_N'].to(device[0]),
            'image_E': batch['image_E'].to(device[0]),
            'image_S': batch['image_S'].to(device[0]),
            'image_W': batch['image_W'].to(device[0])
        }
        output = model(input)
        output_mo_denorm = dataset.denormalize_output(output.clone().detach())
        result_df = pd.concat([
            result_df, 
            pd.DataFrame({
                'uuid': batch['uuid'],
                'mo_latitude': output_mo_denorm[:, 0],
                'mo_longitude': output_mo_denorm[:, 1]
            })], ignore_index=True)

    return result_df



if __name__ == "__main__":
    main(sys.argv[1:])