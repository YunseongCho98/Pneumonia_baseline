import os
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import PneumoniaClassifier
import matplotlib.pyplot as plt

from dataset import PneumoniaDataset
from sklearn.model_selection import train_test_split


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    df = pd.read_csv("pneumonia_images/pneumonia_labels.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = PneumoniaDataset(train_df, "pneumonia_images", transform, from_dataframe=True, mode="train")
    val_dataset = PneumoniaDataset(val_df, "pneumonia_images", transform, from_dataframe=True, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = PneumoniaClassifier().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loss_list = []
    val_loss_list = []

    for epoch in range(10):
        print(f"\nEpoch {epoch+1}/{10}")
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training", leave=True):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_list.append(train_loss / len(train_loader))
        print(f"Epoch {epoch+1} Training_loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=True):
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total if total > 0 else 0
        val_loss_list.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1} Validation_loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), "pneumonia_model.pth")

    # Loss curve 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("모델 학습 완료 및 loss_curve.png 저장 완료")


if __name__ == "__main__":
    main()
