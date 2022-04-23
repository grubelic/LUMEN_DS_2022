# LUMEN_DS_2022
Marry Dijan team's repository for Lumen Data science 2022.

# model3

Implementation of `architecture` module and `main.py` script.

Install `architecture` module in development mode:
`cd model3/src`
`pip install -e .`


## Creating a new Trainer
Trainer refers to a class which has `.train()` method implemented which dictates the training process.
`TrainerDefault` is the default training process.

For creating new Trainer class follow these steps:
1) Create a new directory inside src/architecture/train directory. This directory will be the parrent dir for similar Trainer classes (for example those that differ only by hyperparameters).
2) Create new .py file inside the directory from step 1.
3)* Inside the created file from step 2., create a new class with name signature `Trainer[details]`, for example `Trainer_Classification_DatesetNESW_v1`.
4) Let the created class inherit from default trainer: `Trainer_Classification_DatesetNESW_v1(architecture.train.TrainerDefault)`
5) Update all the hyperparameters and modify the training process.
6)* Declare the newly created class by adding the following line inside `architecture.train.__init__.py`: `from ..... import Trainer_Classification_DatesetNESW_v1`. This step is very important in order that the `main.py` will be able to find the correct trainer class.
7)* Choose a new trainer by specifying `--trainer_name Trainer_Classification_DatesetNESW_v1` when invoking `main.py` script.

Steps marked with * probably cannot be skipped.


