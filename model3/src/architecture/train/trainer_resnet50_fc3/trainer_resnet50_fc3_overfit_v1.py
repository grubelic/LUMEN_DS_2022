from architecture.train import TrainerDefault
from architecture.model.model_resnet50_fc3 import Net
from torchvision import transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

class Trainer_ResNet50_FC3_Overfit_v1(TrainerDefault):
    model = Net()
    # Training parameters
    learning_rate = 0.01
    epoch_num = 100
    batch_size = 2
    batches_per_step = 4

    num_workers = 2

    # Training moderators
    optimizer = SGD
    optimizer_kwds = {
        'momentum': 0.0,
        'weight_decay': 0
    }

    scheduler = StepLR
    scheduler_kwds = {
        'step_size': 10,
        'gamma': 0.1
    }
    dataset_train_kwds = {
        'transform': transforms.Compose([
            #transforms.RandomResizedCrop((224, 224), scale=(0.25, 1.00)),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N', 'E', 'S', 'W']
    }
    dataset_val_kwds = {
        'transform': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'image_directions': ['N', 'E', 'S', 'W']
    }

    def initialize(self):
        super().initialize()

