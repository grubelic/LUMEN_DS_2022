import torch
import torch.nn as nn
import torchvision

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = torchvision.models.resnet50(pretrained=True)
        # Extracting number of features needs to take place before removing 
        # the FC layer from feature extractor.
        num_features = self.feature_extractor.fc.in_features
        # Removing the FC layer from feature extractor.
        self.feature_extractor.fc = nn.Identity()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(num_features*4, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 2)
        )

    def freeze_part(self, model_part, unfreeze=False):
        """
        Use this method to freeze or unfreeze some part of the model.
        Freeze parameters of certain part of the model.
        Set unfreeze=True when you want to unfreeze that part.
        """
        for param in model_part.parameters():
            param.requires_grad = unfreeze

    # Defining the forward pass    
    def forward(self, input):
        features = [
            self.feature_extractor(input['image_N']),
            self.feature_extractor(input['image_E']),
            self.feature_extractor(input['image_S']),
            self.feature_extractor(input['image_W'])
        ]
        features = torch.cat(features, dim=1)
        input = self.fully_connected_layers(features)
        return input