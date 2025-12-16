import torch.nn as nn
from transformers import BlipModel


class BlipForClassification(nn.Module):
    def __init__(self, model_name, num_classes, dropout_prob=0.1):
        super(BlipForClassification, self).__init__()

        # 加载 HuggingFace 预训练模型
        self.blip = BlipModel.from_pretrained(model_name)

        # 获取文本 hidden size (BLIP Base 通常是 768)
        hidden_size = self.blip.config.text_config.hidden_size

        # 分类头设计
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 初始化分类头权重
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, pixel_values, attention_mask):
        # BLIP 前向传播
        # 注意：这里会同时运行 Vision Model 和 Text Model (Cross Attention)
        outputs = self.blip(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_dict=True
        )

        # --- 修复点 ---
        # BlipOutput 对象包含 'text_outputs' 和 'vision_outputs'
        # 我们需要的是融合了图像信息的文本特征，它位于 text_outputs.last_hidden_state 中
        # text_outputs.last_hidden_state shape: [batch, seq_len, 768]

        multimodal_emb = outputs.text_outputs.last_hidden_state[:, 0, :]  # 取 [CLS] token

        # 分类
        x = self.dropout(multimodal_emb)
        logits = self.classifier(x)

        return logits
