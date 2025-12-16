import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ProductDataset(Dataset):
    def __init__(self, csv_path, img_dir, processor, config, mode='train'):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode
        self.config = config

        # 定义图像增强 (训练集增强，验证/测试集仅标准化)
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                # BLIP 预训练时的均值和方差
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((config['data']['img_size'], config['data']['img_size'])),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. 读取图片
        # 假设 id 为 123 -> 图片名为 123.jpg
        img_name = f"{row['id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        # 应用视觉增强
        pixel_values = self.transform(image)

        # 2. 处理文本 (合并 title 和 description)
        # fillna防止空值报错
        title = str(row['title']) if pd.notna(row['title']) else ""
        desc = str(row['description']) if pd.notna(row['description']) else ""
        text_input = f"{title} {desc}".strip()

        # 文本 Tokenize
        text_encoding = self.processor(
            text=text_input,
            padding="max_length",
            truncation=True,
            max_length=self.config['data']['text_max_len'],
            return_tensors="pt"
        )

        input_ids = text_encoding['input_ids'].squeeze(0)
        attention_mask = text_encoding['attention_mask'].squeeze(0)

        item = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'id': row['id']
        }

        # 3. 处理标签
        if self.mode != 'test':
            label = int(row['categories'])
            item['labels'] = torch.tensor(label, dtype=torch.long)

        return item
