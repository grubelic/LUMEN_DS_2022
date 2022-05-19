from sklearn.preprocessing import KernelCenterer
import torch
import torch.nn as nn
import torchvision

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=7)
        )

        self.fully_connected_layers_per_image = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
        )

        self.fully_connected_layers_final = nn.Sequential(
            nn.Linear(256*4, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def freeze_part(self, model_part, unfreeze=False):
        """
        Use this method to freeze or unfreeze some part of the model.
        Freeze parameters of certain part of the model.
        Set unfreeze=True when you want to unfreeze that part.
        """
        for param in model_part.parameters():
            param.requires_grad = unfreeze
    

    def get_grad_norm(self, module):
        parameters = [
            p for p in module.parameters() 
                if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            return None
        device = parameters[0].grad.device
        return torch.stack([
            torch.norm(p.grad.detach()).to(device) for p in parameters])


    # Defining the forward pass    
    def forward(self, input):
        features = [
            self.feature_extractor(input['image_N']),
            self.feature_extractor(input['image_E']),
            self.feature_extractor(input['image_S']),
            self.feature_extractor(input['image_W'])
        ]
        features = [torch.flatten(feat, 1) for feat in features]
        fc_per_image = [
            self.fully_connected_layers_per_image(feat) for feat in features
        ]
        fc_per_image = torch.cat(fc_per_image, dim=1)
        out = self.fully_connected_layers_final(fc_per_image)
        return out