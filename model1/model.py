import torch
import torch.nn as nn

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()

        out_channels1 = 10
        out_channels2 = 20
        out_channels3 = 30

        out_channels4 = 20

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, out_channels1, kernel_size=5, stride=5),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels1, out_channels2, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_channels2),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels2, out_channels3, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_channels3),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(out_channels3 * 1 * 1, out_channels4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels4, 2)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x