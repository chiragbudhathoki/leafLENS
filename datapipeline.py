import torch
import torchvision 
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

transform = v2.Compose([
    v2.Resize((128,128)),
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p= 0.5),
    v2.RandomRotation(degrees=30),
    v2.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.1),
    v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    v2.ToTensor(),
    v2.ToDtype(torch.float32,scale = True),
    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

batch_size = 16

dataset = ImageFolder(root = r'E:\Ml\projects\leafLENS\PlantVillage',transform = transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size, shuffle = True)

