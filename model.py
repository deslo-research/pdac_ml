import torch.nn as nn
import torchvision.models as models
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    num_classes: int = 3
    model_name: str = 'resnet18'  # ['resnet18', 'resnet34', 'resnet50']
    pretrained: bool = True
    dropout_rate: float = 0.5
    hidden_size: int = 128
    freeze_layers: int = 6
    input_size: tuple = (518, 518) 
    activation: str = 'relu'  # ['relu', 'gelu', 'silu']

class CustomModel(nn.Module):
    """Generic model class that supports different backbone architectures"""
    def __init__(self, config: ModelConfig):
        super(CustomModel, self).__init__()
        self.config = config
        
        self.model_dict = {
            'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
            'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
            'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }
        
        self._initialize_backbone()
        self._initialize_classifier()
        self._freeze_layers()
        self._initialize_weights()

    def _initialize_backbone(self):
        """Initialize the backbone network"""
        if self.config.model_name not in self.model_dict:
            raise ValueError(f"Model {self.config.model_name} not supported. "
                           f"Choose from {list(self.model_dict.keys())}")
        
        model_class, weights = self.model_dict[self.config.model_name]
        self.features = model_class(weights=weights if self.config.pretrained else None)
        
        if self.config.model_name.startswith('resnet'):
            self.num_features = self.features.fc.in_features
            self.features.fc = nn.Identity()

    def _initialize_classifier(self):
        """Initialize the classifier head"""
        if self.config.activation == 'relu':
            activation = nn.ReLU
        elif self.config.activation == 'gelu':
            activation = nn.GELU
        elif self.config.activation == 'silu':
            activation = nn.SiLU
        else:
            raise ValueError(f"Activation {self.config.activation} not supported")

        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.num_features, self.config.hidden_size),
            nn.BatchNorm1d(self.config.hidden_size, momentum=0.01),
            activation(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_size, self.config.num_classes)
        )

    def _freeze_layers(self):
        # If config is set to None, freeze all layers
        if self.config.freeze_layers is None:
            for param in self.features.parameters():
                param.requires_grad = False
        elif self.config.freeze_layers > 0:
            for i, child in enumerate(self.features.children()):
                if i < self.config.freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False
        

    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_all_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_layer_info(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}, Trainable: {param.requires_grad}")

    @property
    def device(self):
        return next(self.parameters()).device