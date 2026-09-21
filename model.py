import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self,num_classes= 15):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)

        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(in_features,num_classes )



    def forward(self, x):
        return self.resnet(x)

# if __name__ == "__main__":
#     model = Net()

#     dummyset = torch.randn(16,3,128,128)

#     print("Testing Forward Pass")
#     output = model(dummyset)
#     print(f"The shape of the input : {dummyset.shape}")
#     print(f'The shape of the output: {output.shape}')

    
        



        
