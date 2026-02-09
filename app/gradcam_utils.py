from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

def generate_gradcam(model, input_tensor, image, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(np.argmax(model(input_tensor).cpu().detach().numpy()))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam
