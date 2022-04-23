from architecture.train.trainer_default import TrainerDefault
from architecture.model.debug_model import Net
from torchvision import transforms

class Trainer_DebugVisualization(TrainerDefault):
    """
    This Trainer class is used for dataset visualization.
    BE AWARE: transforms.Normalize is a neccesarry transformation for ResNet
    """
    def train(self):
        self.model = Net()
        self.dataset_train_kwds = {
            'transform': transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.25, 1.00)),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'image_directions': ['N', 'E', 'S', 'W']
        }
        self.dataset_val_kwds = {
            'transform': transforms.Compose([
                transforms.Resize((224, 224)),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'image_directions': ['N', 'E', 'S', 'W']
        }
        self.initialize()
        self.visualize_dataset()

