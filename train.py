import os
import random
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.SimpleCNN import SimpleCNN
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

# -----------------------------
# 可视化函数
# -----------------------------
def plot_metrics(log_dir, train_accuracies, val_accuracies, train_losses, val_losses):
    """绘制训练和验证的准确率和损失曲线，并保存到指定目录
    Args:
        log_dir (str): 日志目录，用于保存图像
        train_accuracies (list): 训练准确率列表
        val_accuracies (list): 验证准确率列表
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表 
    """
    plt.figure()
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/accuracy_plot.png")
    plt.close()

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{log_dir}/loss_plot.png")
    plt.close()

# -----------------------------
# 评估函数
# -----------------------------
def evaluate_model(model, val_loader, criterion, device):
    """评估模型在验证集上的性能
    Args:
        model (nn.Module): 训练好的模型
        val_loader (DataLoader): 验证集数据加载器
        criterion (nn.Module): 损失函数
        device (torch.device): 设备（CPU或GPU）
    Returns:
        accuracy (float): 验证集准确率
        avg_loss (float): 平均损失
        cm (np.ndarray): 混淆矩阵
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, avg_loss, cm

# -----------------------------
# 训练函数
# -----------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, device, log_dir, model_save_path, num_epochs=30, patience=5):
    """
    训练模型并在验证集上评估性能，使用TensorBoard记录训练过程
    Args:
        model (nn.Module): 训练的模型
        train_loader (DataLoader): 训练集数据加载器
        val_loader (DataLoader): 验证集数据加载器
        criterion (nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        device (torch.device): 设备（CPU或GPU）
        log_dir (str): TensorBoard日志目录
        model_save_path (str): 模型保存路径
        num_epochs (int): 训练轮数
        patience (int): 早停策略的耐心值
    """
    writer = SummaryWriter(log_dir)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_acc, val_loss, _ = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} - "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        plot_metrics(log_dir, train_accuracies, val_accuracies, train_losses, val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("Validation loss improved. Model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in {patience} epochs.")
                break

    writer.close()

# -----------------------------
# 主程序
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train FER2013 classifier with configurable model and input")
    parser.add_argument('--model_type', type=str, default='simplecnn', choices=['simplecnn', 'resnet18'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--num_input_channels', type=int, default=3)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--use_tiny_dataset', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = './FER2013'
    model_save_path = f'./checkpoints/{args.model_type}_fer2013.pth'
    log_dir = f'./logs/fer2013_{args.model_type}'

    print(f"Using device: {device}")
    print(f"Model: {args.model_type}, Image Size: {args.image_size}, Channels: {args.num_input_channels}")
    if args.use_tiny_dataset:
        print("Using tiny dataset for quick training")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=args.num_input_channels),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * args.num_input_channels, std=[0.5] * args.num_input_channels)
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'test'), transform=transform)

    if args.use_tiny_dataset:
        train_dataset = torch.utils.data.Subset(train_dataset, random.sample(range(len(train_dataset)), 1000))
        val_dataset = torch.utils.data.Subset(val_dataset, random.sample(range(len(val_dataset)), 1000))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    # 模型定义
    if args.model_type == 'simplecnn':
        model = SimpleCNN(num_classes=7).to(device)
    else:
        model_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }
        model_class = model_dict[args.model_type]
        if args.pretrained_path:
            print(f"Loading pretrained {args.model_type} from {args.pretrained_path}")
            # 先加载预训练模型
            pretrained_dict = torch.load(args.pretrained_path, map_location=device)
            
            # 创建新模型
            model = model_class(pretrained=False)
            model.load_state_dict(pretrained_dict, strict=False)

            # 修改输出层
            model.fc = nn.Linear(model.fc.in_features, 7)
        model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, log_dir, model_save_path, num_epochs=30, patience=5)
    print("Training finished.")
    
    # 在model_save_path上评估模型
    # 自定义标签
    emotion_labels = {
        0: "angry",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise"
    }

    # 在model_save_path上评估模型
    model.load_state_dict(torch.load(model_save_path))
    val_acc, val_loss, cm = evaluate_model(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {val_acc:.2f}%, Final Validation Loss: {val_loss:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 画出混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # 使用自定义标签
    tick_marks = range(len(emotion_labels))
    label_names = [emotion_labels[i] for i in tick_marks]
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"{log_dir}/confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved.")

    

if __name__ == "__main__":
    main()