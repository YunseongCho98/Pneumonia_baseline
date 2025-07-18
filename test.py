import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from model import PneumoniaClassifier
from dataset import PneumoniaDataset


# 하이퍼파라미터 설정
label_csv = "pneumonia_test_images/pneumonia_test_labels.csv"
image_dir = "pneumonia_test_images"
model_path = "pneumonia_model.pth"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
output_csv = "pneumonia_test_results.csv"


def main():
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 데이터셋 로드
    test_dataset = PneumoniaDataset("pneumonia_test_images/pneumonia_test_labels.csv", "pneumonia_test_images", transform, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 로딩
    model = PneumoniaClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 추론 및 결과 저장
    all_preds = []
    all_labels = []
    filenames = []

    with torch.no_grad():
        for images, labels, fnames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            filenames.extend(fnames)

    # 결과 평가
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    # 예측 결과 저장
    result_df = pd.DataFrame({
        "filename": filenames,
        "true_label": all_labels,
        "predicted_label": all_preds
    })
    result_df.to_csv(output_csv, index=False)
    print(f"\n예측 결과 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()
