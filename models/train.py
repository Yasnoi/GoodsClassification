import torch
import torch.optim as optim
from torch import nn
import os

from utils.data_loader import data_loader
from models.net import Net


class TrainNet(nn.Module):
    def __init__(self, config):
        super(TrainNet, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = config['model']['epochs']
        self.learning_rate = config['model']['learning_rate']
        self.weight_decay = config['model']['weight_decay']

        self.train_data_loader, self.val_data_loader = data_loader(self.config, mode='train')

        self.model = Net().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def train_epoch(self, data_loader):
        self.model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, data in enumerate(data_loader):
            images = data['images'].to(self.device)
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            target = data['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images, input_ids, attention_mask)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()

            total_samples += target.shape[0]

            if batch_idx % 100 == 0:
                batch_accuracy = 100. * correct_predictions / total_samples
                print(f'Batch: {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}, Current Accuracy: {batch_accuracy:.2f}%')

        epoch_loss = epoch_loss / total_samples
        epoch_accuracy = 100. * correct_predictions / total_samples

        return epoch_loss, epoch_accuracy

    def validate_epoch(self, data_loader):
        self.model.eval()
        validate_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for data in data_loader:
                images = data['images'].to(self.device)
                input_ids = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                target = data['labels'].to(self.device)

                outputs = self.model(images, input_ids, attention_mask)
                loss = self.criterion(outputs, target)
                validate_loss += loss.item() * images.size(0)

                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()
                total_samples += target.shape[0]

        val_loss = validate_loss / total_samples
        val_accuracy = 100. * correct_predictions / total_samples

        return val_loss, val_accuracy

    def train(self):
        for epoch in range(self.epochs):
            print(f'------------------Start training epoch {epoch + 1}------------------')
            train_loss, train_accuracy = self.train_epoch(self.train_data_loader)
            print(f'------------------Start validating epoch {epoch + 1}------------------')
            val_loss, val_accuracy = self.validate_epoch(self.val_data_loader)
            print(f'Epoch: {epoch + 1}/{self.epochs}\n'
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n'
                  f'Validate Loss: {val_loss:.4f}, Validate Accuracy: {val_accuracy:.2f}%')