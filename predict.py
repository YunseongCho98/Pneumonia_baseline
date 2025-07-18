import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import PneumoniaClassifier
from dataset import UnlabeledImageDataset

# 하이퍼파라미터
image_dir = "samples"
model_path = "pneumonia_model.pth"
output_csv = "samples/sample_predictions.csv"
batch_size = 1

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # 이미지 전처리 transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 데이터셋 및 DataLoader
    dataset = UnlabeledImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 모델 로딩
    model = PneumoniaClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 추론 수행
    predictions = []
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1).cpu().tolist()
            for fname, pred in zip(filenames, preds):
                predictions.append((fname, pred))

    # 결과 저장
    df = pd.DataFrame(predictions, columns=["filename", "predicted_label"])
    df.to_csv(output_csv, index=False)
    print(f"예측 결과 저장 완료: {output_csv} ({len(df)}개)")


if __name__ == "__main__":
    main()