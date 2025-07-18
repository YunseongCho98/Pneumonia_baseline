import os
import pandas as pd
import shutil

# 경로 설정
label_csv = "pneumonia_labels.csv"
image_dir = "pneumonia_images"
test_dir = "pneumonia_test_images"
os.makedirs(test_dir, exist_ok=True)

# CSV 불러오기
df = pd.read_csv(label_csv)

# ✅ 폐렴/비폐렴 각각 31개씩 샘플링
pneumonia_test = df[df['label'] == 1].sample(n=31, random_state=42)
normal_test = df[df['label'] == 0].sample(n=31, random_state=42)
test_df = pd.concat([pneumonia_test, normal_test]).reset_index(drop=True)

# ✅ 테스트 이미지 복사
for fname in test_df['filename']:
    src = os.path.join(image_dir, fname)
    dst = os.path.join(test_dir, fname)
    if os.path.isfile(src):
        shutil.move(src, dst)  # 이동
    else:
        print(f"❌ 파일 없음: {fname}")

# ✅ 테스트 CSV 저장
test_df.to_csv("pneumonia_test_labels.csv", index=False)
print(f"✅ pneumonia_test_labels.csv 저장 완료 (총 {len(test_df)}개)")

# ✅ 원래 라벨에서 테스트에 빠진 것만 남기기
remaining_df = df[~df['filename'].isin(test_df['filename'])].reset_index(drop=True)
remaining_df.to_csv(label_csv, index=False)
print(f"✅ 기존 {label_csv} 갱신 완료 (남은 수: {len(remaining_df)})")
