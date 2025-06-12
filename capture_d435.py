import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 建立資料夾
os.makedirs('dataset/rgb', exist_ok=True)
os.makedirs('dataset/depth', exist_ok=True)

# 初始化相機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

frame_id = 0
print("🔸 按下 S 鍵儲存照片，Q 鍵離開")

try:
    while True:
        frames = pipeline.wait_for_frames()
        print(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 顯示畫面
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        both = np.hstack((color_image, depth_vis))
        cv2.imshow("Color | Depth", both)

        key = cv2.waitKey(1)
        if key == ord('s'):
            rgb_path = f'dataset/rgb/{frame_id:04d}.png'
            depth_path = f'dataset/depth/{frame_id:04d}.png'
            cv2.imwrite(rgb_path, color_image)
            cv2.imwrite(depth_path, depth_image)
            print(f"✅ Saved {frame_id}")
            frame_id += 1
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

