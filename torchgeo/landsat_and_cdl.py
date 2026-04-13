from pathlib import Path

from torch.utils.data import DataLoader
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import Landsat8, Landsat9, CDL
from utils.plots import plot_sat_image_and_mask


# 공통 밴드 인덱스 의미: (Blue, Green, Red, NIR, SWIR1, SWIR2)
common_bands = ["B2", "B3", "B4", "B5", "B6", "B7"]

landsat8 = Landsat8(paths="./data/Landsat8", bands=common_bands)
landsat9 = Landsat9(paths="./data/Landsat9", bands=common_bands)
landsat = landsat8 | landsat9

# 2023년도 CDL(Cropland Data Layer) 데이터셋 다운로드 및 로드
cdl = CDL(paths="./data/CDL", download=True, checksum=True, years=[2023])
dataset = landsat & cdl

sampler = RandomGeoSampler(dataset, size=512, length=10000)  # 512x512 크기의 랜덤 샘플링, 길이 10,000개의 샘플링
dataloader = DataLoader(dataset=dataset, batch_size=8, sampler=sampler)

for batch in dataloader:
    image = batch["image"]  # shape: [batch_size, channels, height, width]
    mask = batch["mask"]    # shape: [batch_size, height, width]
    
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")

    plot_sat_image_and_mask(
        img=image[0], mask=mask[0],
        cmap=CDL.cmap,  # CDL 클래스 팔레트를 호출 측에서 주입 (plots.py는 dataset-agnostic)
        save_file_prefix="sample",
        rgb_bands=(2, 1, 0),
    )
        
    # 여기서 모델 학습이나 추론을 수행할 수 있습니다
    # 예: model(image) 또는 loss = criterion(model(image), mask)
    
    # 첫 번째 배치만 확인하고 종료
    break