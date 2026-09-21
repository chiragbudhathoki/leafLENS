import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from model import Net

# 1. Page Configuration
st.set_page_config(
    page_title="leafLENS – Is this leaf healthy?",
    page_icon="🌿",
    layout="centered"
)

# 2. Load Model & Pipeline (Cached so it only loads once)
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("leaflens.pth", map_location=device, weights_only=True))
    model.eval()
    return model, device

model, device = load_model()

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 3. UI Layout
st.title("🌿 leafLENS")
st.write("Upload a photo of a tomato, potato, or bell pepper leaf to check its health status.")

uploaded_file = st.file_uploader("Choose a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    
    if st.button("Analyze Leaf", type="primary"):
        with st.spinner("Reading the leaf..."):
            # Preprocess and predict
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            conf_score = confidence.item() * 100
            
            # Confidence threshold check (rejection protocol)
            if conf_score < 60.0:
                st.warning(f"**Result: Unrecognized / Not a recognized leaf** (Confidence: {conf_score:.2f}%)")
                st.info("The model wasn't sure enough. Try uploading a clearer, closer photo of a single leaf.")
            else:
                st.success(f"**Prediction:** {predicted_class}")
                st.metric(label="Confidence Score", value=f"{conf_score:.2f}%")