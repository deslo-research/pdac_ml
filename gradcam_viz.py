from pytorch_grad_cam import GradCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from model import ModelConfig, CustomModel

def get_class_label(class_idx: int) -> str:
    if isinstance(class_idx, torch.Tensor):
        class_idx = class_idx.item()
    return "Cancer" if class_idx == 0 else "Fibrosis/Normal"

def run_gradcam(
        model_path: str,
        model_config: ModelConfig,
        dataset,
        indices_path: str,
        device: torch.device,
        results_dir: str = 'gradcam_results') -> None:

    test_indices = torch.load(indices_path)
    
    model = CustomModel(model_config).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if model_config.model_name.startswith('resnet'):
        target_layers = [model.features.layer4[-1]]
    else:
        raise ValueError(f"Unsupported model architecture: {model_config.model_name}")

    cam = FullGrad(model=model, target_layers=target_layers)
    
    os.makedirs(results_dir, exist_ok=True)
    
    images = [dataset[i][2] for i in test_indices[:5]]  # First 5 test images
    labels = [dataset[i][3] for i in test_indices[:5]]
    print(labels)
    
    for i, image in enumerate(images):
        input_tensor = image.to(device)
        grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0))[0, :]
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        cam_image = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(image_np)
        plt.title(f'Image: {get_class_label(labels[i])}')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title('FullGradCAM Heatmap')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(cam_image)
        plt.title('FullGradCAM Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'gradcam_sample_{i}.png'))
        plt.close()

def visualize_gradcam_with_indices(model_path, model_config, dataset, device):
    indices_path = model_path.replace('best.pth', 'test_indices.pt')
    
    run_gradcam(
        model_path=model_path,
        model_config=model_config,
        dataset=dataset,
        indices_path=indices_path,
        device=device,
        results_dir='gradcam_results'
    )