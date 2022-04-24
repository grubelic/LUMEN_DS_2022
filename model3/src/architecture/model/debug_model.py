import torch
import torch.nn as nn
import torchvision

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        # Extracting number of features needs to take place before removing 
        # the FC layer from feature extractor.
        num_features = self.feature_extractor.fc.in_features
        # Removing the FC layer from feature extractor.
        self.feature_extractor.fc = nn.Identity()
        # All parameters in feature extractor will be frozen.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    # Defining the forward pass    
    def forward(self, input):
        input = input['image_N']
        input = self.feature_extractor(input)
        input = self.fully_connected_layers(input)
        return input