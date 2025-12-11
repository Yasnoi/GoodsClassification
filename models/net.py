import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class Net(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(Net, self).__init__()

        # Image Encoder
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        # output dim 2048

        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        # output dim 768

        if freeze_backbone:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Fusion Head
        input_dim = 2048 + 768

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 21)
        )

    def forward(self, images, input_ids, attention_mask):
        # Image Stream
        # image size (batch_size, 3, 224, 224)
        img_features = self.image_encoder(images)
        # ResNet output size (batch_size, 2048, 1, 1), a flat is needed
        img_features = img_features.view(img_features.size(0), -1)

        # Text Stream
        # text size (batch_size, seq_len)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        # output size (batch_size, 768)

        combined_features = torch.cat((img_features, text_features), dim=1)

        outputs = self.classifier(combined_features)
        return outputs
