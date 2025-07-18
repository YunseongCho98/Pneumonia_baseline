import torch
import cv2
import numpy as np
from model import PneumoniaClassifier
from torchvision import transforms
from PIL import Image

def grad_cam(model, image_tensor, target_layer):
    gradients = []
    activations = []

    def save_gradients_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations_hook(module, input, output):
        activations.append(output)

    handle1 = target_layer.register_forward_hook(save_activations_hook)
    handle2 = target_layer.register_backward_hook(save_gradients_hook)

    model.eval()
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()
    score = output[0, class_idx]
    score.backward()

    grad = gradients[0].detach().cpu().numpy()[0]
    act = activations[0].detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def show_cam_on_image(img_path, model_path):
    model = PneumoniaClassifier()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)

    cam = grad_cam(model, img_tensor, model.backbone.layer4[1].conv2)

    img_np = np.array(img_pil.resize((224, 224))) / 255.
    cam_img = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    overlay = 0.5 * img_np + 0.5 * (cam_img / 255.)

    # 이미지 저장
    output_path = img_path.replace('.png', '_gradcam.png')
    cv2.imwrite(output_path, np.uint8(255 * overlay))
    print(f"Grad-CAM 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    show_cam_on_image("samples/00004737_006.png", "pneumonia_model.pth")
    show_cam_on_image("samples/00007629_001.png", "pneumonia_model.pth")