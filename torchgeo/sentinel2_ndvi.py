import matplotlib
matplotlib.use("Agg")  # 원격 서버에서 구동 고려
import matplotlib.pyplot as plt
from torchgeo.datasets import Sentinel2
from torchgeo.transforms import AppendNDVI

safe_path = "./data/Sentinel-2.Incheon.SAFE"

dataset = Sentinel2(
    paths=safe_path,
    bands=["B02", "B03", "B04", "B08"])

bounds = dataset.bounds
sample = dataset[bounds]

# Sentinel2 RGB 시각화 저장 
fig = dataset.plot(sample)
fig.savefig("sentinel2_rgb.png", dpi=150, bbox_inches="tight")
plt.close(fig)


# AppendNDVI 사용법
transform = AppendNDVI(index_red=2, index_nir=3)  # B04=2 (Red), B08=3 (NIR)
image_tensor = sample['image']
image_with_ndvi = transform(image_tensor)   # (4, H, W) -> (1, 5, H, W)

batch_0 = image_with_ndvi[0]  # (1, 5, H, W) -> (5, H, W)
ndvi_band = batch_0[-1].cpu().numpy()  # NDVI range [-1, 1]

# NDVI 시각화 저장 (colorbar 포함)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(ndvi_band, cmap="RdYlGn", vmin=-1, vmax=1)
ax.set_title("NDVI")
ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig("sentinel2_ndvi.png", dpi=150, bbox_inches="tight")
plt.close(fig)
