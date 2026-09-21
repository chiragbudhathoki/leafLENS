import os
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import v2 
import torch.nn.functional as F
from model import Net

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict_leaf(image_path,model_path= 'leaflens.pth'):
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load(model_path,map_location = device,weights_only = True))
    model.eval()

    transform= v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32,scale= True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor= transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probability = F.softmax(output[0],dim = 0)
        confidence,predict_idx = torch.max(probability,0)
    predicted_class = CLASS_NAMES[predict_idx.item()]
    confidence_score = confidence.item()*100

    if confidence_score < 50:
        print(f"Result: Unknown / Not a recognized leaf (Confidence: {confidence_score:.2f}%)")
    else:
        print(f"Result: {predicted_class} (Confidence: {confidence_score:.2f}%)")
          
    return predicted_class,confidence_score

if __name__ == "__main__":
    predict_leaf('dog.jpg')