import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
from transformers import BlipProcessor

# 引入自定义模块
from src.utils.common import load_config, setup_logger, seed_everything
from src.utils.dataset import ProductDataset
from src.models.blip_cls import BlipForClassification


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # 混合精度训练 (A800 必备)
        with autocast():
            logits = model(input_ids, pixel_values, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Cosine Annealing 每个 batch 更新

        total_loss += loss.item()
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Val]", leave=False):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                logits = model(input_ids, pixel_values, attention_mask)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    # 1. Setup
    config = load_config("src/configs/config.yaml")
    logger = setup_logger(config['output']['dir'])
    seed_everything(config['train']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device} (Should be A800)")

    # 2. Data Preparation
    logger.info("Loading Data...")
    processor = BlipProcessor.from_pretrained(config['train']['model_name'])

    full_dataset = ProductDataset(
        csv_path=config['data']['train_csv'],
        img_dir=config['data']['train_img_dir'],
        processor=processor,
        config=config,
        mode='train'
    )

    # 9:1 划分验证集
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 修改验证集的 transform 为非增强模式 (利用 dataset.py 中的 mode 逻辑需要一点小技巧，
    # 但简单起见，random_split 后的 subset 会继承父类属性。
    # 为了严谨，建议在 val loop 里不依赖 dataset 的 transform，或者在这里重新实例化 val_ds，
    # 但由于 ProductDataset 在 init 时根据 mode 决定 transform，这里 split 后的 subset
    # 依然会使用 train 的 transform。
    # **修正方案**：为了最强性能，我们重新实例化验证集 Dataset。

    # 重新定义验证集 Dataset 以确保不使用数据增强
    # 注意：这里需要根据索引手动切分，或者简单地使用两个Dataset对象
    # 简单做法：我们接受验证集也有一点点增强，或者使用 subset 的 indices 重构
    # 既然有 A800，我们追求完美：
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = torch.utils.data.Subset(
        ProductDataset(config['data']['train_csv'], config['data']['train_img_dir'], processor, config, mode='train'),
        train_indices
    )
    val_ds = torch.utils.data.Subset(
        ProductDataset(config['data']['train_csv'], config['data']['train_img_dir'], processor, config, mode='val'),
        val_indices
    )

    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True,
                              num_workers=config['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False,
                            num_workers=config['data']['num_workers'], pin_memory=True)

    # 3. Model Initialization
    logger.info("Initializing Model...")
    model = BlipForClassification(
        model_name=config['train']['model_name'],
        num_classes=config['data']['num_classes']
    ).to(device)

    # 4. Optimizer & Scheduler & Loss
    optimizer = AdamW(model.parameters(), lr=config['train']['learning_rate'],
                      weight_decay=config['train']['weight_decay'])

    # 标签平滑 CrossEntropy
    criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'])

    # 余弦退火调度器
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scaler = GradScaler()  # 混合精度

    # 5. Training Loop
    best_acc = 0.0
    best_model_path = os.path.join(config['output']['dir'], config['output']['best_model_name'])

    logger.info("Start Training...")
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device,
                                                epoch + 1)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            f"Epoch {epoch + 1}/{config['train']['epochs']} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with Acc: {best_acc:.4f}")

    # 6. Inference / Testing
    logger.info("Start Inference on Test Set...")
    # 加载最佳权重
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_ds = ProductDataset(
        csv_path=config['data']['test_csv'],
        img_dir=config['data']['test_img_dir'],
        processor=processor,
        config=config,
        mode='test'
    )
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False,
                             num_workers=config['data']['num_workers'])

    results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']

            with autocast():
                logits = model(input_ids, pixel_values, attention_mask)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for id_val, pred in zip(ids, preds):
                results.append({'id': id_val, 'categories': pred})  # 注意列名要匹配提交要求

    # 生成提交文件
    submission_path = os.path.join(config['output']['dir'], config['output']['submission_name'])
    df_sub = pd.DataFrame(results)
    df_sub.to_csv(submission_path, index=False)
    logger.info(f"Inference Done. Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
