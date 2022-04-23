from architecture.train.trainer_default import TrainerDefault
from architecture.model.debug_model import Net
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

class Trainer_Debug_V1(TrainerDefault):
    model = Net()
    # Training parameters
    learning_rate = 0.0005
    epoch_num = 100
    batch_size = 100
    batches_per_step = 1

    # Training moderators
    optimizer = SGD
    optimizer_kwds = {
        'momentum': 0.1,
        'weight_decay': 0
    }

    scheduler = StepLR
    scheduler_kwds = {
        'step_size': 20,
        'gamma': 0.5
    }
    dataset_train_kwds = {
        'transform': transforms.Compose([
            #transforms.RandomResizedCrop((224, 224), scale=(0.25, 1.00)),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N']
    }
    dataset_val_kwds = {
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N']
    }


