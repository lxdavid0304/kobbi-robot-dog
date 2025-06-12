
import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

# 自動讀取桌面最新的 .raw/.csv/.png
desktop = Path.home() / "OneDrive" / "Desktop"
raw_path = max(desktop.glob("*_Depth.raw"), key=os.path.getmtime)
csv_path = max(desktop.glob("*_Depth_metadata.csv"), key=os.path.getmtime)
rgb_path = max(desktop.glob("*_Color.png"), key=os.path.getmtime)

print("📂 RAW:", raw_path)
print("📂 CSV:", csv_path)
print("📂 RGB:", rgb_path)

# 以純 Python 處理 metadata（不使用 pandas）
metadata = {}
with open(csv_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 2:
            key = parts[0].strip().lower()
            val = parts[1].strip()
            metadata[key] = val

def get_value(key):
    key = key.lower()
    for k in metadata:
        if key in k:
            return float(metadata[k])
    raise ValueError(f"❌ 無法找到欄位: {key}")

width = int(get_value("resolution x"))
height = int(get_value("resolution y"))
fx = get_value("fx")
fy = get_value("fy")
ppx = get_value("ppx")
ppy = get_value("ppy")

depth = np.fromfile(raw_path, dtype=np.uint16).reshape((height, width))
cv2.imwrite("depth_restored.png", depth)

color_raw = cv2.imread(str(rgb_path))
color_resized = cv2.resize(color_raw, (width, height))

color_o3d = o3d.geometry.Image(cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB))
depth_o3d = o3d.geometry.Image(depth)
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, ppx, ppy)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

if len(pcd.points) > 0:
    o3d.io.write_point_cloud("pointcloud_from_raw.ply", pcd)
    print("✅ 已儲存點雲：pointcloud_from_raw.ply")
    o3d.visualization.draw_geometries([pcd])
else:
    print("⚠️ 無有效點雲點")
