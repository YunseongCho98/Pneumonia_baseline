
# Pneumonia Diagnosis Classifier (ResNet18 기반)

이 프로젝트는 흉부 X‑ray 이미지를 기반으로 **폐렴 여부를 분류**하는 딥러닝 모델입니다.  
PyTorch와 torchvision의 **ResNet‑18** 백본을 사용하며, Grad‑CAM으로 시각적 설명도 제공합니다.

---

## 📁 프로젝트 구조

```
project_root/
├── train.py               # 모델 학습
├── test.py                # 테스트 데이터 평가
├── predict.py             # 라벨 없는 샘플 이미지 추론
├── gradcam.py             # Grad‑CAM 시각화
├── model.py               # ResNet18 기반 분류기 정의
├── dataset.py             # 커스텀 Dataset 클래스
├── pneumonia_model.pth    # 학습된 모델 파라미터 (학습 후 생성)
├── loss_curve.png         # 학습/검증 손실 그래프  (학습 후 생성)
├── samples/               # 예측용 샘플 이미지
├── pneumonia_images/      # 학습 이미지 & 라벨(csv)
├── pneumonia_test_images/ # 테스트 이미지 & 라벨(csv)
└── README.md
```

---

## 🧠 모델 개요

* **백본**: ResNet‑18  
* **출력 노드 수**: 2 (`No Finding`, `Pneumonia`)  
* **손실 함수**: Cross‑Entropy  
* **최적화 기법**: Adam (lr = 1e‑4)  

---

## 🔧 사용 방법

### 1) 모델 학습
```bash
python train.py
```
* `pneumonia_images/pneumonia_labels.csv`를 8 : 2 비율로 train/val 분할  
* 학습 완료 후 **`pneumonia_model.pth`**, **`loss_curve.png`** 생성

### 2) 모델 평가
```bash
python test.py
```
* `pneumonia_test_images/` 데이터셋 평가  
* 결과는 **`pneumonia_test_results.csv`** 에 저장  
* 콘솔에 confusion matrix 및 classification report 출력

### 3) 샘플 예측 (무라벨 이미지)
```bash
python predict.py
```
* `samples/` 폴더 이미지 예측 → **`samples/sample_predictions.csv`** 저장

### 4) Grad‑CAM 시각화
```bash
python gradcam.py
```
* `samples/` 이미지에 대한 Grad‑CAM 생성  
* `_gradcam.png` 접미사로 저장

---

## 📄 CSV 포맷

### 학습/테스트 라벨 예시
```csv
filename,label
00000013_008.png,0   # 0 = No Finding
00000013_009.png,1   # 1 = Pneumonia
```

---

## 🛠️ 의존성

| 패키지 | 버전(예시) |
| ------ | ---------- |
| python | >= 3.8 |
| torch | >= 1.10 |
| torchvision | >= 0.11 |
| pandas | >= 1.3 |
| scikit‑learn | >= 1.0 |
| pillow | >= 8.0 |
| tqdm | >= 4.60 |
| matplotlib | >= 3.4 |
| opencv‑python | >= 4.5 |

설치:
```bash
pip install torch torchvision pandas pillow scikit-learn tqdm matplotlib opencv-python
```

---

## 🔍 특징

* **ResNet‑18** 기반 간단·경량 모델  
* **Confusion Matrix / Classification Report** 출력  
* **Grad‑CAM** 으로 관심 영역 시각화  
* **라벨 없는 이미지 배치 예측** 기능

---

## 📜 라이선스

본 프로젝트는 MIT License를 따릅니다.

---

## 📬 문의

이슈나 PR로 질문 및 제안을 남겨주세요.
