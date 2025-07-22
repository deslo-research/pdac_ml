import numpy as np
import os
from PIL import Image
import re
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom
import random
import torch
import torchvision.transforms.functional as TF
from collections import Counter

def custom_collate_fn(batch):
    auto_fluorescent = [item[0] for item in batch]
    shg = [item[1] for item in batch]
    combined_image = torch.stack([item[2] for item in batch])
    labels = torch.tensor([item[3] for item in batch])
    
    return auto_fluorescent, shg, combined_image, labels

def combine_images(channel_1, channel_2, florecent_scaler=1.1):
    channel_1 = channel_1.astype(float)
    channel_2 = channel_2.astype(float)
    
    c1_max = channel_1.max()
    c2_max = channel_2.max()
    
    # Normalize
    if c1_max > 0:
        channel_1 = channel_1 * (255.0 / c1_max)
    if c2_max > 0:
        channel_2 = channel_2 * (255.0 / c2_max)
    
    combined_image = np.zeros((channel_1.shape[0], channel_1.shape[1], 3))
    
    combined_image[..., 1] = np.clip(channel_2 * florecent_scaler, 0, 255)  # green
    combined_image[..., 2] = channel_1  # blue
    
    return combined_image.astype(np.uint8)

def generate_synthetic_sample(auto_fluorescent, shg, target_size=None):
    auto_fluorescent = auto_fluorescent.astype(np.float32)
    shg = shg.astype(np.float32)
    
    auto_max = auto_fluorescent.max()
    shg_max = shg.max()
    
    auto_fluorescent = auto_fluorescent / auto_max if auto_max > 0 else auto_fluorescent
    shg = shg / shg_max if shg_max > 0 else shg
    
    angle = random.uniform(-30, 30)
    scale = random.uniform(0.8, 1.2)
    
    aug_auto_fluorescent = rotate(auto_fluorescent, angle, reshape=False, order=1)
    aug_shg = rotate(shg, angle, reshape=False, order=1)
    
    if target_size is not None:
        h, w = target_size
        current_h, current_w = aug_auto_fluorescent.shape
        scale_h = h / current_h
        scale_w = w / current_w
        scale = min(scale_h, scale_w)
    
    aug_auto_fluorescent = zoom(aug_auto_fluorescent, scale, order=1)
    aug_shg = zoom(aug_shg, scale, order=1)
    
    noise_level = random.uniform(0, 0.1)
    aug_auto_fluorescent += np.random.normal(0, noise_level, aug_auto_fluorescent.shape)
    aug_shg += np.random.normal(0, noise_level, aug_shg.shape)
    
    aug_auto_fluorescent = np.clip(aug_auto_fluorescent, 0, 1)
    aug_shg = np.clip(aug_shg, 0, 1)
    
    aug_auto_fluorescent = (aug_auto_fluorescent * auto_max).astype(np.uint16)
    aug_shg = (aug_shg * shg_max).astype(np.uint16)
    
    if target_size is not None:
        aug_auto_fluorescent = TF.resize(Image.fromarray(aug_auto_fluorescent), target_size)
        aug_shg = TF.resize(Image.fromarray(aug_shg), target_size)
        aug_auto_fluorescent = np.array(aug_auto_fluorescent)
        aug_shg = np.array(aug_shg)
    
    return aug_auto_fluorescent, aug_shg

class PancreaticCancerDataset(Dataset):
    def __init__(self, cancer_path, healthy_path, transform=None, balance_dataset=False, use_normal=False, target_size=(512, 512)):
        self.transform = transform
        self.samples = []
        self.synthetic_samples = []
        self.balance_dataset = balance_dataset
        self.target_size = target_size
        self.use_normal = use_normal
        
        self.florecent_scaler = 1.0
        self.process_path(cancer_path, label=1)  # 1 for cancer
        self.process_path(healthy_path, label=0)  # 0 for healthy
        
        if self.balance_dataset:
            self.generate_synthetic_samples()

    def process_path(self, path, label):
        for patient in os.listdir(path):
            patient_path = os.path.join(path, patient)
            if os.path.isdir(patient_path):
                if label == 1: 
                    for subfolder in os.listdir(patient_path):
                        subfolder_path = os.path.join(patient_path, subfolder)
                        if os.path.isdir(subfolder_path):
                            subfolder_label = self.get_subfolder_label(subfolder)
                            if subfolder_label != -1:
                                self.process_patient_folder(subfolder_path, subfolder_label)
                else:
                    self.process_patient_folder(patient_path, label)

    def get_subfolder_label(self, subfolder_name):
        if subfolder_name.lower() == 'cancer':
            return 1
        elif subfolder_name.lower() == 'fibrosis':
            return 0
        elif subfolder_name.lower() in ['normal adjacent', 'normal']:
            return 0 if self.use_normal else 2
        else:
            print(f"Unknown subfolder type: {subfolder_name}. Treating as normal (0).")
            return 0

    def process_patient_folder(self, folder_path, label):
        files = os.listdir(folder_path)
        
        file_groups = {}
        for file in files:
            if file.endswith('.tif'):
                match = re.search(r'area(\d+)', file)
                if match:
                    area_number = int(match.group(1))
                    base_name = file[:match.start()]
                    if base_name not in file_groups:
                        file_groups[base_name] = {}
                    file_groups[base_name][area_number] = file
                else:
                    print(f"Skipping file with unexpected format: {file}")

        for base_name, group in file_groups.items():
            area_numbers = sorted(group.keys())
            for i in range(len(area_numbers) - 1):
                if area_numbers[i] + 1 == area_numbers[i+1]:
                    auto_fluorescent = group[area_numbers[i]]
                    shg = group[area_numbers[i+1]]
                    
                    auto_fluorescent_path = os.path.join(folder_path, auto_fluorescent)
                    shg_path = os.path.join(folder_path, shg)
                    self.samples.append((auto_fluorescent_path, shg_path, label))

    def generate_synthetic_samples(self):
        """Generate synthetic samples to balance the dataset"""
        label_counts = self.get_label_distribution()
        max_samples = max(label_counts.values())
        
        for label, count in label_counts.items():
            if count < max_samples:
                samples_needed = max_samples - count
                label_samples = [(auto_path, shg_path) for auto_path, shg_path, lbl in self.samples if lbl == label]
                
                for _ in range(samples_needed):
                    auto_path, shg_path = random.choice(label_samples)
                    
                    auto_fluorescent = np.array(Image.open(auto_path))
                    shg = np.array(Image.open(shg_path))
                    
                    aug_auto_fluorescent, aug_shg = generate_synthetic_sample(
                        auto_fluorescent, shg, target_size=self.target_size
                    )
                    
                    combined_image = combine_images(aug_auto_fluorescent, aug_shg, self.florecent_scaler)
                    self.synthetic_samples.append((aug_auto_fluorescent, aug_shg, combined_image, label))

    def __len__(self):
        return len(self.samples) + len(self.synthetic_samples)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            auto_fluorescent_path, shg_path, label = self.samples[idx]
            auto_fluorescent = np.array(Image.open(auto_fluorescent_path))
            shg = np.array(Image.open(shg_path))
            
            if self.target_size:
                auto_fluorescent = np.array(TF.resize(Image.fromarray(auto_fluorescent), self.target_size))
                shg = np.array(TF.resize(Image.fromarray(shg), self.target_size))
            
            combined_image = combine_images(auto_fluorescent, shg, self.florecent_scaler)
            combined_image = Image.fromarray(combined_image)
        else:
            auto_fluorescent, shg, combined_image, label = self.synthetic_samples[idx - len(self.samples)]
            combined_image = Image.fromarray(combined_image)
        
        if self.transform:
            combined_image = self.transform(combined_image)
        
        return auto_fluorescent, shg, combined_image, label

    def get_label_distribution(self):
        labels = [label for _, _, label in self.samples]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))
    
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_data_splits(dataset, config):
    """Create train and test splits with fixed seed"""
    set_seed(config.seed)
    
    n_samples = len(dataset)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    
    test_split = int(np.floor(config.test_size * n_samples))
    train_val_indices = indices[test_split:]
    test_indices = indices[:test_split]
    
    labels = [dataset[i][3].item() if torch.is_tensor(dataset[i][3]) 
             else dataset[i][3] for i in range(len(dataset))]
    
    class_counts = Counter(labels)
    print("Full dataset class distribution:", class_counts)
    
    test_labels = [labels[i] for i in test_indices]
    print("Test set class distribution:", Counter(test_labels))
    
    train_val_labels = [labels[i] for i in train_val_indices]
    print("Train/Val set class distribution:", Counter(train_val_labels))
    
    return train_val_indices, test_indices