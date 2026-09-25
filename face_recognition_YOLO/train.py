import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import FaceCNNModel

# Config
DATA_DIR = "data/dataset_cnn"
MODEL_SAVE_PATH = "data/best_model.pth"
BENCHMARK_SAVE_PATH = "data/cnn_benchmark.csv"
BATCH_SIZE = 32
EPOCHS = 35
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def evaluate_benchmark(model, val_loader, val_dataset, device, model_save_path):
    """
    Tải weights tốt nhất và xuất các chỉ số Benchmark chi tiết.
    """
    print("\n" + "=" * 65)
    print("      ĐANG TIẾN HÀNH ĐÁNH GIÁ BENCHMARK TRÊN TẬP VALIDATION")
    print("=" * 65)

    # 1. Load trọng số tốt nhất
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"-> Đã tải thành công trọng số tốt nhất từ: {model_save_path}")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    # 2. Đo tốc độ suy luận (Inference Speed)
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        # Warmup GPU
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        for _ in range(10):
            _ = model(dummy_input)

        for images, labels in val_loader:
            images = images.to(device)
            
            # Bắt đầu đo thời gian inference
            start_time = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_time
            
            total_time += batch_time
            total_samples += images.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    # 3. Tính toán các chỉ số Benchmark
    acc = accuracy_score(all_targets, all_preds)
    prec_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    avg_inference_ms = (total_time / total_samples) * 1000  # ms/ảnh
    fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0

    # 4. Tạo DataFrame Benchmark
    benchmark_data = [{
        "Model": "FaceCNNModel",
        "Total Val Samples": total_samples,
        "Num Classes": len(val_dataset.classes),
        "Accuracy": round(acc, 4),
        "Precision (Weighted)": round(prec_weighted, 4),
        "Recall (Weighted)": round(rec_weighted, 4),
        "F1-Score (Weighted)": round(f1_weighted, 4),
        "F1-Score (Macro)": round(f1_macro, 4),
        "Inference Speed (ms/img)": round(avg_inference_ms, 2),
        "Throughput (FPS)": round(fps, 2)
    }]

    df_benchmark = pd.DataFrame(benchmark_data)

    # In kết quả dạng bảng
    print("\n--- BẢNG KẾT QUẢ BENCHMARK ---")
    for col, val in df_benchmark.iloc[0].items():
        print(f"{col:<25}: {val}")
    print("=" * 65)

    # Lưu ra CSV
    df_benchmark.to_csv(BENCHMARK_SAVE_PATH, index=False)
    print(f"-> Đã lưu bảng Benchmark vào file: '{BENCHMARK_SAVE_PATH}'\n")

# Khối bảo vệ multiprocessing trên Windows
if __name__ == '__main__':
    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Init Model, Loss, Optimizer
    model = FaceCNNModel(num_classes=len(train_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Bắt đầu huấn luyện CNN trên thiết bị: {DEVICE}\n" + "=" * 65)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_top1 = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc1, _ = accuracy(outputs, labels, topk=(1, 5))
            train_loss += loss.item() * images.size(0)
            train_top1 += acc1 * images.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)
        train_top1 /= len(train_dataset)

        # Validation
        model.eval()
        val_loss, val_top1, val_top5 = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                val_loss += loss.item() * images.size(0)
                val_top1 += acc1 * images.size(0)
                val_top5 += acc5 * images.size(0)

        val_loss /= len(val_dataset)
        val_top1 /= len(val_dataset)
        val_top5 /= len(val_dataset)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc1: {train_top1:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc1: {val_top1:.2f}% Acc5: {val_top5:.2f}%")

        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> Đã lưu mô hình tốt nhất với Val Top-1 Accuracy: {val_top1:.2f}%")

    print("=" * 65 + f"\nHuấn luyện hoàn tất! Trọng số lưu tại: {MODEL_SAVE_PATH}")

    # --- ĐẦU RA BENCHMARK TỰ ĐỘNG SAU HUẤN LUYỆN ---
    evaluate_benchmark(model, val_loader, val_dataset, DEVICE, MODEL_SAVE_PATH)