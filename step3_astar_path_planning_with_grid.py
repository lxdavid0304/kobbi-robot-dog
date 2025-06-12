
import numpy as np
import cv2
import heapq

# === 載入佔用地圖 ===
grid = np.load("occupancy_map_256.npy")
h, w = grid.shape

# === 起點與終點格子座標（可修改）===
start = (250, 20)  # 格式為 (row=z, col=x)
goal = (20, 235)

# === 鄰居方向（八連通）===
neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    visited = set()
    pq = [(0 + heuristic(start, goal), 0, start, [])]
    while pq:
        f, g, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        path = path + [node]
        if node == goal:
            return path
        visited.add(node)
        for dx, dy in neighbors:
            x2, y2 = node[0] + dx, node[1] + dy
            if 0 <= x2 < h and 0 <= y2 < w and grid[x2, y2] == 0:
                heapq.heappush(pq, (g + 1 + heuristic((x2, y2), goal), g + 1, (x2, y2), path))
    return []

# === 執行 A* ===
path = astar(grid, start, goal)

# === 畫圖像 ===
img = (1 - grid) * 255
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for y, x in path:
    cv2.circle(img_color, (x, y), 1, (0, 0, 255), -1)

cv2.circle(img_color, (start[1], start[0]), 4, (0, 255, 255), -1)
cv2.circle(img_color, (goal[1], goal[0]), 4, (0, 255, 0), -1)

# === 加上網格座標與座標軸 ===
font = cv2.FONT_HERSHEY_SIMPLEX
grid_size = 256
x_min, x_max = -1.5, 1.5
z_min, z_max = 0.0, 3.0
x_res = (x_max - x_min) / grid_size
z_res = (z_max - z_min) / grid_size

for i in range(0, grid_size + 1, 64):
    x_val = x_min + i * x_res
    z_val = z_max - i * z_res
    cv2.line(img_color, (i, 0), (i, grid_size), (180, 180, 180), 1)
    cv2.line(img_color, (0, i), (grid_size, i), (180, 180, 180), 1)
    cv2.putText(img_color, f"{x_val:.1f}", (i+2, grid_size - 5), font, 0.3, (100, 100, 255), 1)
    cv2.putText(img_color, f"{z_val:.1f}", (2, i - 2), font, 0.3, (0, 100, 0), 1)

cv2.putText(img_color, "X: -1.5m ~ 1.5m", (5, 12), font, 0.4, (0, 0, 255), 1)
cv2.putText(img_color, "Z: 0.0m ~ 3.0m", (5, 26), font, 0.4, (0, 0, 255), 1)

cv2.imwrite("occupancy_astar_path_grid.png", img_color)
print("✅ 路徑圖已輸出：occupancy_astar_path_grid.png")
cv2.imshow("A* Path with Grid", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
