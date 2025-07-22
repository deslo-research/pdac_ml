import torch
import os
from torchvision import transforms

from model import ModelConfig, CustomModel
from train import TrainingConfig, train_model
from dataset import PancreaticCancerDataset

from eval import evaluate_with_indices
from gradcam_viz import visualize_gradcam_with_indices
import argparse

def best_model_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "/home/ares/Data/Pancreatic SHG Images"
    healthy_path = os.path.join(data_path, "Healthy Areas")
    cancer_path = os.path.join(data_path, "Cancer & Fibrosis Areas")

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])
    dataset = PancreaticCancerDataset(cancer_path, healthy_path, balance_dataset=False, use_normal=True, transform=transform)

    model_type = "cancer_resnet18_binary_new"
    # model_type = "healthy_resnet18_se_2_class"

    model_path = f"models/{model_type}/model_best.pth"
    # This should to be the same as the trained model parameters
    model_config = ModelConfig(
        num_classes=2,
        model_name='resnet18', 
        pretrained=True,
        dropout_rate=0.1,
        hidden_size=512,
        freeze_layers=5,
        activation="relu"
    )
    evaluate_with_indices(model_path, model_config, dataset, device)
    visualize_gradcam_with_indices(model_path, model_config, dataset, device)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((518, 518)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    # data_path = "C:/Users/deslo/Documents/Data/PancreaticSlideImagesAnnotated/Pancreatic SHG Images"
    data_path = "/home/ares/Data/Pancreatic SHG Images"
    healthy_path = os.path.join(data_path, "Healthy Areas")
    cancer_path = os.path.join(data_path, "Cancer & Fibrosis Areas")
    dataset = PancreaticCancerDataset(cancer_path, healthy_path, balance_dataset=False, use_normal=True, transform=transform)

    # Setup for model configuration
    model_config = ModelConfig(
        num_classes=2,
        model_name='resnet18', 
        pretrained=True,
        dropout_rate=0.1,
        hidden_size=512,
        freeze_layers=2,
        activation="relu"
    )

    training_config = TrainingConfig(
        batch_size=32,
        num_epochs=100,
        base_lr=1e-4,
        weight_decay=0.01,
        label_smoothing=0.1,
        warmup_epochs=2,
        patience=10,
        num_folds=5,
        test_size=0.2,
        mixup_alpha=0.2,
        grad_clip=1,
        seed=42
    )

    model = CustomModel(model_config)

    results = train_model(
        model=model,
        dataset=dataset,
        config=training_config,
        device=device,
        model_save_path='models/cancer_resnet18_binary_news/model'
    )

    print("Training Results Summary:")
    print(f"Best Model Path: {results['best_model_path']}")
    print(f"Best Fold: {results['best_fold']}")
    print("Best Metrics:", {k: f"{v:.4f}" for k, v in results['best_metrics'].items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    if args.eval_only:
        print("Skipping training, eval only")
        best_model_eval()
    else:
        main()
        best_model_eval()