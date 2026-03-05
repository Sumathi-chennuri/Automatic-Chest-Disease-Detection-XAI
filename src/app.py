import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 14)  # 14 diseases
    model.eval()
    model.to(device)
    return model
model = load_model()
classes = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
    "Emphysema","Fibrosis","Hernia","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Pleural Thickening"
]
st.title("🫁 Automatic Chest Disease Detection")
st.write(
    "Upload a Chest X-ray image to detect possible thoracic diseases "
    "using Deep Learning and Explainable AI (Grad-CAM)."
)
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = torch.sigmoid(model(input_tensor))[0].cpu().numpy()
    threshold = 0.5
    affected_indices = np.where(outputs >= threshold)[0]
    if len(affected_indices) == 0:
        st.success("✅ Lungs appear NORMAL")
    else:
        st.error("⚠️ Lungs are AFFECTED")
        st.write("### Detected Disease(s):")
        for i in affected_indices:
            st.write(f"- **{classes[i]}** : {outputs[i]*100:.2f}%")
    if len(affected_indices) > 0:
        st.subheader("Grad-CAM Heatmaps")
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        top_indices = affected_indices[
            np.argsort(outputs[affected_indices])[::-1][:2]
        ]
        for i in top_indices:
            targets = [ClassifierOutputTarget(i)]
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]
            img_np = np.array(image.resize((224, 224))) / 255.0
            heatmap = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
            st.write(f"### {classes[i]}")
            st.image(heatmap, use_container_width=True)
    st.success("✅ Analysis Complete!")
