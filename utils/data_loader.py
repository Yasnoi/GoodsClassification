import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class GoodsDataset(Dataset):
    def __init__(self, data_frame, img_dir, transform, tokenizer, max_len=512):
        self.data_frame = data_frame.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.data_frame['title'] = self.data_frame['title'].fillna('')
        self.data_frame['description'] = self.data_frame['description'].fillna('')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        # Image Processing
        # get the image
        img_name = str(row['id']) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        # read the image
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, FileNotFoundError):
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        # apply image transform
        if self.transform:
            image = self.transform(image)

        # Text Processing
        # get text description
        text_raw = str(row['title']).strip() + ' ' + str(row['description']).strip()
        # convert text to tensor
        encoding = self.tokenizer.encode_plus(
            text_raw,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Label Processing
        # get the label
        label = int(row['categories'])

        # return data as Dict
        return {
            'images': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def data_loader(config, mode='train'):
    csv_path = os.path.join(config['data']['data_dir'], config['data'][f'{mode}_csv_file'])
    image_path = os.path.join(config['data']['data_dir'], config['data'][f'{mode}_image_dir'])

    batch_size = config['model']['batch_size']

    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('data/distilbert_local')

    df = pd.read_csv(csv_path)

    if mode == 'train':
        # define Transforms
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # split train data and validate data
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['categories'])
        # instantiate dataset
        train_set = GoodsDataset(train_df, image_path, train_transform, tokenizer)
        val_set = GoodsDataset(val_df, image_path, val_transform, tokenizer)
        # instantiate dataloader
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
        return train_loader, val_loader
    elif mode == 'test':
        # define Transforms
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        # instantiate dataset
        test_set = GoodsDataset(df, image_path, test_transform, tokenizer)
        # instantiate dataloader
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
        return test_loader
