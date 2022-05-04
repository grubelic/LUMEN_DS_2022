# LUMEN_DS_2022
Marry Dijan team's repository for Lumen Data science 2022.

# model3

Implementation of `architecture` module, `train.py` and `inference.py` scripts.

## Setup

The easiest and most straightforward way to set things up is using Docker.

To install Docker, follow these instructions: https://docs.docker.com/get-docker/

Now build a Docker image from Dockerfile:

`docker build -t marry_dijan -f model3/Docker/Dockerfile_cpu model3/Docker/`

Now run new container and mount your workspace directory to container's workspace
directory:

`docker run -it --rm -v [FULL PATH TO YOUR WORKSPACE]:/workspaces marry_dijan /bin/bash`

Now a new bash instance should be running inside the created container.

Install `architecture` module in development mode:
```
cd LUMEN_DS_2022/model3/src
pip install -e .
```

Now you are ready to run either `train.py` or `inference.py` from `model3/src`.
Usage instructions are explained below.


## Creating a new Trainer
Trainer refers to a class which has `.train()` method implemented which dictates the training process.
`TrainerDefault` is the default training process.

For creating new Trainer class follow these steps.
Let's suppose that we want to create a training process for a neural network
we have just implemented inside `model_resnet34_fc11.py` file. We are not sure
what hyperparameters to use, so we will create two training processes that 
differ only in hyperparameters. Each training process gets it's own `.py` file, 
and those training processes that differ only in hyperparameters should be 
inside the same parent directory.

1) Create parent directory `src/architecture/train/trainers_resnet34_fc11/` and
inside this directory create files `trainer_resnet34_fc11_v1.py` and
`trainer_resnet34_fc11_v2.py`.

2) \* Inside each of these files create create a new class with name signature 
`Trainer[details]`, in our case `Trainer_ResNet34_FC11_v1` and `Trainer_ResNet34_FC11_v2` respectively.

3) Let the created classes inherit from default trainer: 
   
   `class Trainer_ResNet34_FC11_v1(architecture.train.TrainerDefault)`

4) Update all the hyperparameters and modify the training process.
5) \* Declare the newly created class by adding the following line inside `architecture.train.__init__.py`: 
   
   `from .trainers_resnet34_fc11.trainer_resnet34_fc11_v1 import Trainer_ResNet34_FC11_v1`. 
   
   This step is very important in order that `main.py` would be able to find the correct trainer class.

6)  \* Choose a new trainer by specifying `--trainer_name Trainer_ResNet34_FC11_v1` when invoking `main.py` script.

Steps marked with * probably cannot be skipped.

## Running inference
The term 'inference' means generating predictions of the trained model on some
input images. For running inference, run `inference.py` script.

In the following example we will generate predictions of one model.

- Parameters is located at `/workspaces/parameters_id-25.prms`.

- Dataset is located at:

  ```
  /workspaces/Dataset/
      |- data/
      |- 2022-mar-31_data_val.csv
  ```

- (Empty) output directory is located at `/workspaces/id_25_out`.

Now run the following:
```
python3 src/inference.py
   --trainer_name=Trainer_ResNet50_FC3_v1 \
   --parameters_path=/workspaces/parameters_id-25.prms \
   --dataset_root=/workspaces/Dataset \
   --target_csv=2022-mar-31_data_val.csv \
   --output_dir=/workspaces/id_25_out \
   --device=cpu \
   --batch_size=1 \
   --num_workers=4
```

Result is `/workspaces/id_25_out/output.csv` where all the predictions are generated
in columns `mo_latitude`, `mo_longitude`.

# Useful information

To run a process on a remote machine with `ssh`:
`ssh name@host "python3 /path/to/script.py --param1 val1 > /path/to/stdout 2>&1 &"`


# Debugging in VSCode
Add this configuration to `.vscode/launch.json` for a debug training.

```
{
   "name": "Training with train.py",
   "type": "python",
   "request": "launch",
   "program": "src/train.py",
   "console": "integratedTerminal",
   "justMyCode": true,
   "args": [
         "--trainer_name=Trainer_Debug_V1",
         "--output_dir=outputs/debug-output-3",
         "--train_csv=2022-mar-31_data-debug_train.csv",
         "--val_csv=2022-mar-31_data-debug_val.csv",
         "--dataset_root=/workspaces/Dataset",
         "--device=cpu",
         "--run_mode=d"
   ]
}
```
