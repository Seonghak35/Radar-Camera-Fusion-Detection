import os
import numpy as np
import pandas as pd
from PIL import Image

# ✅ 데이터셋 경로 설정
dataset_path = 'data'
os.makedirs(dataset_path, exist_ok=True)

# ✅ 디렉토리 생성
dirs = ['image', 'radar', 'detection']
for d in dirs:
    os.makedirs(os.path.join(dataset_path, d), exist_ok=True)

# ✅ 가짜 이미지 생성
def create_dummy_image(file_path, size=(640, 480)):
    image = Image.fromarray(np.uint8(np.random.rand(*size, 3) * 255))
    image.save(file_path)

# ✅ 가짜 레이더 데이터 생성
def create_dummy_radar_data(file_path, num_points=100):
    data = {
        'timestamp': np.random.rand(num_points),
        'range': np.random.rand(num_points) * 100,
        'doppler': np.random.randn(num_points),
        'azimuth': np.random.rand(num_points) * 360,
        'elevation': np.random.rand(num_points) * 180,
        'power': np.random.rand(num_points) * 50,
        'x': np.random.randn(num_points),
        'y': np.random.randn(num_points),
        'z': np.random.randn(num_points),
        'u': np.random.rand(num_points) * 640,
        'v': np.random.rand(num_points) * 480,
        'label': np.random.randint(0, 7, num_points),
        'instance': np.random.randint(0, 10, num_points)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# ✅ 가짜 라벨 (YOLO 형식 바운딩 박스) 생성
def create_dummy_label(file_path, num_objects=5):
    with open(file_path, "w") as f:
        for _ in range(num_objects):
            class_id = np.random.randint(0, 7)  # 0~6 클래스
            x_center = np.random.rand()  # 0~1 범위 (640 기준)
            y_center = np.random.rand()  # 0~1 범위 (480 기준)
            width = np.random.rand() * 0.2  # 0~0.2 (전체 이미지 대비)
            height = np.random.rand() * 0.2
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# ✅ 데이터셋 개수 설정
num_samples = 50

# ✅ 더미 데이터 생성
for i in range(num_samples):
    create_dummy_image(f"{dataset_path}/image/{i:06d}.jpg")
    create_dummy_radar_data(f"{dataset_path}/radar/{i:06d}.csv")
    create_dummy_label(f"{dataset_path}/detection/{i:06d}.txt")

print("✅ 가짜 WaterScenes 데이터 & YOLO Labels 생성 완료!")
