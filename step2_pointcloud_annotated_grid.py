
import numpy as np
import cv2
import open3d as o3d

# === 載入點雲 ===
pcd = o3d.io.read_point_cloud("pointcloud_from_raw.ply")
points = np.asarray(pcd.points)

if len(points) == 0:
    print("⚠️ 無有效點雲，請先執行 step1")
    exit()

# === 儲存世界座標 ===
np.savetxt("points.csv", points, delimiter=",", header="x,y,z", comments='')
print("✅ 已輸出世界座標：points.csv")

# === 根據 Z 軸距離排序並選取最近/最遠的障礙物 ===
z_sorted = points[points[:, 2].argsort()]
top5_near = z_sorted[:5]
top5_far = z_sorted[-5:]

# === 印出最近與最遠障礙物座標 ===
print("\n📌 最近的 5 筆障礙物 (Z 最小)：")
for i, pt in enumerate(top5_near):
    print(f"  {i+1}. x={pt[0]:.3f}, y={pt[1]:.3f}, z={pt[2]:.3f} m")

print("\n📌 最遠的 5 筆障礙物 (Z 最大)：")
for i, pt in enumerate(top5_far):
    print(f"  {i+1}. x={pt[0]:.3f}, y={pt[1]:.3f}, z={pt[2]:.3f} m")

# === 建立棋盤格子圖 ===
grid_size = 256
x_min, x_max = -1.5, 1.5
z_min, z_max = 0.0, 3.0
x_res = (x_max - x_min) / grid_size
z_res = (z_max - z_min) / grid_size
grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

def project_to_grid(pt):
    x, _, z = pt
    gx = int((x - x_min) / x_res)
    gz = int((z_max - z) / z_res)
    return gx, gz

for pt in points:
    gx, gz = project_to_grid(pt)
    if 0 <= gx < grid_size and 0 <= gz < grid_size:
        grid[gz, gx] = 1

# === 建立圖像並加上標記 ===
img = (1 - grid) * 255
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# 繪製網格與刻度
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(0, grid_size + 1, 64):
    x_val = x_min + i * x_res
    z_val = z_max - i * z_res
    cv2.line(img_color, (i, 0), (i, grid_size), (180, 180, 180), 1)
    cv2.line(img_color, (0, i), (grid_size, i), (180, 180, 180), 1)
    cv2.putText(img_color, f"{x_val:.1f}", (i+2, grid_size - 5), font, 0.3, (100, 100, 255), 1)
    cv2.putText(img_color, f"{z_val:.1f}", (2, i - 2), font, 0.3, (0, 100, 0), 1)

cv2.putText(img_color, "X: -1.5m ~ 1.5m", (5, 12), font, 0.4, (0, 0, 255), 1)
cv2.putText(img_color, "Z: 0.0m ~ 3.0m", (5, 26), font, 0.4, (0, 0, 255), 1)

# === 標記近/遠障礙物座標點 ===
for pt in top5_near:
    gx, gz = project_to_grid(pt)
    if 0 <= gx < grid_size and 0 <= gz < grid_size:
        cv2.circle(img_color, (gx, gz), 4, (0, 255, 255), -1)  # 黃色：最近

for pt in top5_far:
    gx, gz = project_to_grid(pt)
    if 0 <= gx < grid_size and 0 <= gz < grid_size:
        cv2.circle(img_color, (gx, gz), 4, (255, 0, 255), -1)  # 粉紅：最遠

cv2.imwrite("occupancy_map_annotated.png", img_color)
print("\n✅ 已儲存圖：occupancy_map_annotated.png")
cv2.imshow("Occupancy Map with Annotations", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
