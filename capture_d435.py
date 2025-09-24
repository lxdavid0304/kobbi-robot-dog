# -*- coding: utf-8 -*-
"""
Open3D 版｜逐步偵錯輸出（每一步都 print）
- 4m x 4m ROI：x,z ∈ [-2, 2]；GRID_SIZE=320（~1.25cm/格）
- 半徑 0.5 m「朝前半圓」安全足跡（中心 z=-1.5）
- 三視窗：Color / Depth / Occupancy(live)
- C：背景存 color/depth/pointcloud(用 Open3D) 到桌面（只傳 numpy 副本）
- P：以目前地圖跑 A*，桌面輸出路徑圖＋CSV
- Q：離開
"""
import os, sys, time, platform, traceback
from math import ceil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as pkg_version, PackageNotFoundError

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d

# ================== 全域設定 ==================
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")

GRID_SIZE = 320
ROI = dict(x_min=-2.0, x_max=2.0, z_min=-2.0, z_max=2.0)

ROBOT_X_W = 0.0
ROBOT_Z_W = -1.5
ROBOT_RADIUS_M = 0.20
HALFCIRCLE_RADIUS_M = 0.50
HALFCIRCLE_CENTER_OFFSET_M = 0.0

MAP_UPDATE_INTERVAL_SEC = 0.2
DRAW_GRID_ANNOTATION = False

A_STAR_START_WORLD = (ROBOT_X_W, ROBOT_Z_W)
A_STAR_GOAL_WORLD  = (0.0, 1.5)

POINTS_STRIDE = 1  # Open3D 已做取樣/濾波，這裡保留 1（全像素）

# --------- 小工具：統一風格的 log ----------
def _now():
    return time.strftime("%H:%M:%S")

def log(step, msg=None, *, flush=True):
    # 允許 log("message") 或 log("step", "message")
    if msg is None:
        step, msg = "misc", step
    print(f"[{_now()}] [{step}] {msg}", flush=flush)

def log_ok(step="misc"):
    print(f"[{_now()}] [{step}] ✅ OK", flush=True)

def log_fail(step, e):
    print(f"[{_now()}] [{step}] ❌ FAIL: {type(e).__name__}: {e}", flush=True)

def log_exc(step):
    log(step, "❌ EXCEPTION ↓↓↓\n" + traceback.format_exc())
    
def debug_env():
    step = "debug_env"
    try:
        try:
            rs_ver = pkg_version("pyrealsense2")
        except PackageNotFoundError:
            rs_ver = "unknown"
        cv_ver  = getattr(cv2, "__version__", "unknown")
        o3d_ver = getattr(o3d, "__version__", "unknown")

        log(step, "===== 環境資訊 =====")
        log(step, f"OS: {platform.platform()}")
        log(step, f"Python: {sys.version.split()[0]}")
        log(step, f"OpenCV: {cv_ver}")
        log(step, f"pyrealsense2: {rs_ver}")
        log(step, f"Open3D: {o3d_ver}")
        log(step, "====================")

        # 壓低 Open3D 日誌
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        log_ok(step)
    except Exception:
        log_exc(step); raise

def check_frames_match(color_np, depth_np, intr):
    Hc, Wc = color_np.shape[:2]
    Hd, Wd = depth_np.shape[:2]
    ok = True

    print(f"[chk] color: {color_np.shape} {color_np.dtype}")
    print(f"[chk] depth: {depth_np.shape} {depth_np.dtype}")
    print(f"[chk] intrinsic: {intr.width}x{intr.height}")

    # 尺寸一致性
    if (Hc, Wc) != (Hd, Wd):
        print(f"[chk] ❌ color 與 depth 尺寸不同: {(Hc,Wc)} vs {(Hd,Wd)}")
        ok = False
    if (intr.width, intr.height) != (Wd, Hd):
        print(f"[chk] ❌ 內參尺寸與影像不符: intr={intr.width}x{intr.height} vs img={Wd}x{Hd}")
        ok = False

    # dtype 合法性
    if color_np.dtype != np.uint8:
        print(f"[chk] ⚠️ color 建議是 uint8，目前是 {color_np.dtype}")
    if depth_np.dtype not in (np.uint16, np.float16, np.float32, np.float64):
        print(f"[chk] ⚠️ depth 非常規 dtype: {depth_np.dtype}")

    # 有沒有 NaN / inf（浮點深度常見）
    if np.issubdtype(depth_np.dtype, np.floating):
        bad = np.isnan(depth_np).sum() + np.isinf(depth_np).sum()
        if bad > 0:
            print(f"[chk] ⚠️ depth 浮點含有 NaN/Inf: {bad} 個")

    # 深度零值比例（0 代表無效距離）
    zeros = (depth_np == 0).sum() if depth_np.dtype != np.bool_ else 0
    total = depth_np.size
    print(f"[chk] depth 零值比例: {zeros}/{total} ({zeros/total:.1%})")

    print("[chk] ✅ 通過" if ok else "[chk] ❌ 不通過（請先修正）")
    return ok
    
def assert_intrinsic_and_images(color_np, depth_np, intr):
    Hc, Wc = color_np.shape[:2]
    Hd, Wd = depth_np.shape[:2]
    Hi, Wi = intr.height, intr.width

    print(f"[chk] color={color_np.shape}, dtype={color_np.dtype}")
    print(f"[chk] depth={depth_np.shape}, dtype={depth_np.dtype}")
    print(f"[chk] intrinsic={Wi}x{Hi}")

    # --- 基本一致性 ---
    assert Hc == Hd and Wc == Wd, \
        f"❌ color 與 depth 尺寸不一致: color={Hc}x{Wc}, depth={Hd}x{Wd}"
    assert Wd == Wi and Hd == Hi, \
        f"❌ depth 尺寸 {Wd}x{Hd} 與 intrinsic {Wi}x{Hi} 不一致"

    # --- dtype 嚴格檢查 ---
    assert color_np.dtype == np.uint8, \
        f"❌ color dtype 應該是 uint8, 拿到 {color_np.dtype}"
    assert depth_np.dtype in (np.uint16, np.float32), \
        f"❌ depth dtype 應該是 uint16(mm) 或 float32(m), 拿到 {depth_np.dtype}"

    # --- 特殊檢查: NaN/Inf ---
    if np.issubdtype(depth_np.dtype, np.floating):
        bad = np.isnan(depth_np).sum() + np.isinf(depth_np).sum()
        assert bad == 0, f"❌ depth 裡面有 NaN/Inf ({bad} 個)"

    print("[chk] ✅ 全部檢查通過")

def report_depth_dtype(depth_np):
    if depth_np.dtype == np.uint16:
        print("[dtype] depth 是 uint16（多半是毫米 mm）")
    elif depth_np.dtype == np.float16:
        print("[dtype] depth 是 float16（精度較低，建議先轉 float32）")
    elif depth_np.dtype == np.float32:
        print("[dtype] depth 是 float32（通常是公尺 m）")
    elif depth_np.dtype == np.float64:
        print("[dtype] depth 是 float64（雙精度，通常不必要）")
    else:
        print(f"[dtype] depth 是 {depth_np.dtype}（非常規類型）")

    # 額外印出最小/最大值（若是浮點，順便猜單位）
    try:
        vmin = float(np.nanmin(depth_np))
        vmax = float(np.nanmax(depth_np))
        print(f"[dtype] 深度值範圍: {vmin:.6f} ~ {vmax:.6f}")
        if np.issubdtype(depth_np.dtype, np.floating):
            # 粗略猜測單位
            if vmax > 10.0: print("[dtype] 這個上限偏大，可能是毫米或未縮放的單位")
            else:           print("[dtype] 上限在幾公尺內，較像『公尺 m』")
    except Exception as _:
        pass

# ================== RealSense 自動啟動 ==================

import pyrealsense2 as rs

def auto_start_realsense(w=640, h=480, fps=30, warmup_frames=30, laser_on=True):
    step = "auto_start_realsense"
    pipeline = None
    try:
        log(step, "建立 context / 查裝置")
        ctx = rs.context()
        devs = ctx.query_devices()
        log(step, f"發現裝置數={len(devs)}")
        if len(devs) == 0:
            raise RuntimeError("找不到 RealSense 裝置")

        pipeline = rs.pipeline()
        config = rs.config()
        # ✅ 使用傳入參數，不要寫死
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

        log(step, "啟動 pipeline")
        profile = pipeline.start(config)

        # 裝置資訊（可選）
        dev = profile.get_device()
        try:
            name = dev.get_info(rs.camera_info.name)
            sn   = dev.get_info(rs.camera_info.serial_number)
            log(step, f"裝置：{name} / S/N: {sn}")
        except Exception:
            pass

        # 開啟投射器（若支援）
        try:
            depth_sensor = dev.first_depth_sensor()
            if laser_on and depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
                log(step, "投射器已啟用")
        except Exception:
            log(step, "投射器設定略過")

        log(step, f"暖機丟 {warmup_frames} 幀")
        for _ in range(int(warmup_frames)):
            pipeline.wait_for_frames()

        align = rs.align(rs.stream.color)
        depth_sensor = dev.first_depth_sensor()
        rs_scale = depth_sensor.get_depth_scale()
        log(step, f"depth_scale={rs_scale}")
        log_ok(step)
        return pipeline, profile, align, rs_scale

    except Exception:
        # ✅ 若已啟動，失敗時幫你停掉，避免占住相機
        try:
            if pipeline is not None:
                pipeline.stop()
        except Exception:
            pass
        log_exc(step)
        raise

# ================== 座標/網格工具 ==================
def meters_to_cell(x, z):
    """世界座標(m) -> 格子(row, col)。row=0 在前方(z_max)，z 越大 row 越小。"""
    x_res = (ROI["x_max"] - ROI["x_min"]) / GRID_SIZE
    z_res = (ROI["z_max"] - ROI["z_min"]) / GRID_SIZE
    col = int((x - ROI["x_min"]) / x_res)
    row = int((ROI["z_max"] - z) / z_res)
    # 夾住避免越界
    col = min(max(col, 0), GRID_SIZE - 1)
    row = min(max(row, 0), GRID_SIZE - 1)
    return row, col

def cell_to_meters(row, col):
    """格子(row, col) -> 世界座標(m)，取格心。"""
    x_res = (ROI["x_max"] - ROI["x_min"]) / GRID_SIZE
    z_res = (ROI["z_max"] - ROI["z_min"]) / GRID_SIZE
    x = ROI["x_min"] + (col + 0.5) * x_res
    z = ROI["z_max"] - (row + 0.5) * z_res
    return x, z

def colorize_depth(depth):
    """
    將深度可視化：支援 uint16(mm) 或浮點(m)。
    這只是顯示用途（不影響幾何）。
    """
    import numpy as np, cv2
    depth = np.asarray(depth)
    if depth.dtype == np.uint16:
        vis = cv2.convertScaleAbs(depth, alpha=0.03)  # 視覺scale
    else:
        # 若是公尺，先轉成近似毫米範圍後再壓縮顯示
        vis = cv2.convertScaleAbs(depth.astype(np.float32) * 1000.0, alpha=0.03)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    
    
def place_human_robot_on_4x4_center_back(ROI, GRID_SIZE, *, radius_m=0.5):
    """
    人(=半圓圓心)：在地圖後緣向前 0.5m 的中線上 (x=0)。
    機器人(=半圓最前端)：在人前方 radius_m（=0.5m）。
    回傳：((human_x, human_z), (robot_x, robot_z)), ((human_row, human_col), (robot_row, robot_col))
    """
    x_min, x_max = ROI["x_min"], ROI["x_max"]
    z_min, z_max = ROI["z_min"], ROI["z_max"]

    human_x = 0.0
    human_z = z_min + 0.5          # 後緣向前 0.5 m
    robot_x = human_x
    robot_z = human_z + radius_m   # 機器人在「前方」= z 增加

    human_rc = meters_to_cell(human_x, human_z)
    robot_rc = meters_to_cell(robot_x, robot_z)

    return ((human_x, human_z), (robot_x, robot_z)), (human_rc, robot_rc)


from math import ceil
import numpy as np

def make_forward_semicircle_kernel(
    x_res, z_res,
    *,
    robot_radius_m=0.18,           # 你的機器人半徑（可改）
    halfcircle_radius_m=0.5,       # ★ 向前半圓半徑 = 0.5 m
    lock_robot_at_tip=True,        # ★ 讓「機器人在半圓最前端」
    halfcircle_offset_m=None,      # （若不鎖最前端，才用這個自訂偏移）
    logger=None                    # 可傳入 log(step,msg)；不傳就安靜
):
    """
    建立「向前半圓 + 機器人身體圓」的 kernel，給 OpenCV 的 dilate 使用。
    座標約定：row 往「下」變大；z 往「前」變大 ⇒ 前方在「上面」，所以取 r <= hr 當上半圓。
    OpenCV 的 anchor 需回傳 (x, y) = (col, row)。
    """
    def _log(msg):
        if logger:
            logger("make_forward_semicircle_kernel", msg)

    # --- 1) 半徑換成格數 ---
    rx = max(1, int(ceil(robot_radius_m      / x_res)))  # X 向機器人半徑（格）
    rz = max(1, int(ceil(robot_radius_m      / z_res)))  # Z 向機器人半徑（格）
    hx = max(1, int(ceil(halfcircle_radius_m / x_res)))  # X 向半圓半徑（格）
    hz = max(1, int(ceil(halfcircle_radius_m / z_res)))  # Z 向半圓半徑（格）

    # --- 2) 決定圓心相對位置（off_cells）---
    # 想要「機器人在最前端」⇒ 圓心在機器人後方 hz 格 ⇒ off_cells = +hz
    if lock_robot_at_tip:
        off_cells = hz
    else:
        off_m = float(halfcircle_offset_m or 0.0)
        off_cells = int(round(off_m / z_res))  # + = 往後，- = 往前

    # --- 3) 畫布上下空間要看 offset 方向來預留，避免圓心跑出畫布 ---
    half_up   = max(rz, hz, max(0, -off_cells))  # 負偏移（往前）要多留「上方」
    half_down = max(rz,      max(0,  off_cells)) # 正偏移（往後）要多留「下方」

    H = half_up + half_down + 1                 # kernel 高度
    W = 2 * max(rx, hx) + 1                     # kernel 寬度
    anchor_row = half_up                        # 機器人中心放在這一列
    anchor_col = W // 2                         # 水平置中

    K = np.zeros((H, W), dtype=np.uint8)

    # --- 4) 畫機器人自身圓（內圈）---
    for r in range(H):
        dz_r = (r - anchor_row) * z_res
        for c in range(W):
            dx_c = (c - anchor_col) * x_res
            if dx_c*dx_c + dz_r*dz_r <= robot_radius_m*robot_radius_m:
                K[r, c] = 1

    # --- 5) 畫向前半圓（外圈，圓心可偏移）---
    hr = anchor_row + off_cells                 # 半圓圓心所在列
    hc = anchor_col
    # 夾住圓心，極端參數也不會整個畫不到
    hr = min(max(hr, 0), H - 1)

    for r in range(H):
        dz_r = (r - hr) * z_res
        for c in range(W):
            dx_c = (c - hc) * x_res
            # 只拿「圓心以上」那半邊（= 前方）
            if r <= hr and (dx_c*dx_c + dz_r*dz_r <= halfcircle_radius_m*halfcircle_radius_m):
                K[r, c] = 1

    anchor = (anchor_col, anchor_row)  # OpenCV: (x, y)

    # （可選）小自檢：機器人是否在半圓最前端
    if lock_robot_at_tip:
        tip_ok = (hr - anchor_row) == hz
        _log(f"K={K.shape}, anchor={anchor}, hr={hr}, hz={hz}, tip_ok={tip_ok}")
    else:
        _log(f"K={K.shape}, anchor={anchor}, hr={hr}, off_cells={off_cells}")

    return K, anchor


# ================== Open3D：RGBD → PCD → 佔據格 ==================
def build_o3d_intrinsic_from_frame(color_frame, expect_w=None, expect_h=None):
    """
    從 RealSense 的 color frame 取得 Open3D 的針孔內參。
    若給 expect_w/h，會多做一次尺寸一致性檢查（可傳 color 影像寬高）。
    """
    step = "build_o3d_intrinsic_from_frame"
    try:
        vs = color_frame.profile.as_video_stream_profile()
        intr = vs.intrinsics  # 有 width/height/fx/fy/ppx/ppy/distortion model

        if expect_w is not None and expect_h is not None:
            if (intr.width, intr.height) != (int(expect_w), int(expect_h)):
                raise ValueError(
                    f"內參尺寸與影像不一致: intr={intr.width}x{intr.height} vs img={expect_w}x{expect_h}"
                )

        ointr = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
        )

        # 小提醒：Open3D 這裡不會用到 RealSense 的畸變參數
        # 如果你的鏡頭畸變比較明顯，可考慮先把 color/depth 做去畸變再建點雲
        try:
            model = getattr(intr, "model", None)
            if model is not None and model != 0:  # 0 通常是 "none"
                log(step, f"⚠️ 注意：distortion model={model}，未去畸變直接當 pinhole 使用")
        except Exception:
            pass

        log(step, f"size=({intr.width},{intr.height}) fx={intr.fx:.2f} fy={intr.fy:.2f} "
                  f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}")
        return ointr
    except Exception:
        log_exc(step)
        raise


# ---------- 主函式（全步驟列印 + 降採樣/去雜訊/ROI） ----------
def frames_to_pointcloud_o3d(
    color_np: np.ndarray,
    depth_np: np.ndarray,                    # 建議：RealSense 原生 uint16 (mm)
    o3d_intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_trunc: float = 2.0,               # 單位 m（可見距離上限）
    ROI: dict | None = None,
    voxel_size: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    step_name: str = "frames_to_pointcloud_o3d(u16)"
):
    step = step_name
    log(step, f"START depth_trunc={depth_trunc}, voxel={voxel_size}, denoise=({nb_neighbors},{std_ratio}), ROI={ROI}")

    # 1) 健檢 (超嚴格)
    sub = "1.validate_inputs"
    try:
        Hc, Wc = color_np.shape[:2]
        Hd, Wd = depth_np.shape[:2]

        log(sub, f"color={color_np.shape} {color_np.dtype} | "
                 f"depth={depth_np.shape} {depth_np.dtype} | "
                 f"intr={o3d_intrinsic.width}x{o3d_intrinsic.height}")

        # shape 檢查
        assert color_np.ndim == 3 and color_np.shape[2] == 3, \
            f"color 應為 (H,W,3)，拿到 {color_np.shape}"
        assert depth_np.ndim == 2, \
            f"depth 應為 (H,W)，拿到 {depth_np.shape}"
        assert (Hc, Wc) == (Hd, Wd), \
            f"color/depth 尺寸不一致: {(Hc,Wc)} vs {(Hd,Wd)}"
        assert (Wd, Hd) == (o3d_intrinsic.width, o3d_intrinsic.height), \
            f"intrinsic 尺寸不符: intr={o3d_intrinsic.width}x{o3d_intrinsic.height}, img={Wd}x{Hd}"

        # dtype 檢查
        assert color_np.dtype == np.uint8, f"color 應為 uint8，拿到 {color_np.dtype}"
        assert depth_np.dtype in (np.uint16, np.float16, np.float32, np.float64), \
            f"depth dtype 不支援: {depth_np.dtype}"

        # NaN/Inf 檢查
        if np.issubdtype(depth_np.dtype, np.floating):
            bad = np.isnan(depth_np).sum() + np.isinf(depth_np).sum()
            assert bad == 0, f"depth 含有 NaN/Inf: {bad} 個"

        # 零值比例（0 = 無效距離）
        zeros = (depth_np == 0).sum()
        ratio = zeros / depth_np.size
        if ratio > 0.5:
            log(sub, f"⚠️ 警告: depth 零值比例過高 {ratio:.1%}")

        # depth_trunc 合法
        assert isinstance(depth_trunc, (int, float)) and depth_trunc > 0, "depth_trunc 應為正數（m）"

        log_ok(sub)
    except Exception:
        log_exc(sub)
        raise

    # 2) 連續記憶體 & BGR→RGB
    sub = "2.to_contiguous_and_rgb"
    try:
        color_np = np.ascontiguousarray(color_np, dtype=np.uint8)
        depth_np = np.ascontiguousarray(depth_np)  # 先保留 dtype，後面再決定路徑
        # Open3D 期望 RGB
        color_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        log_ok(sub)
    except Exception:
        log_exc(sub)
        raise

    # 3) 準備深度：改用 float32 (m) 路徑，比 uint16(mm) 更不挑版本
    sub = "3.prepare_depth_f32_m"
    try:
        # 原本 depth_np 可能是 uint16(mm) or float*
        if depth_np.dtype == np.uint16:
            d32 = depth_np.astype(np.float32) * (1.0 / 1000.0)  # mm → m
        else:
            d32 = depth_np.astype(np.float32, copy=False)

        # 清掉壞值 + 負值；並做截斷
        d32[~np.isfinite(d32)] = 0.0
        np.maximum(d32, 0.0, out=d32)
        np.minimum(d32, float(depth_trunc), out=d32)

        # ★ 強制成連續且對齊的 buffer，避免底層 C++ 因對齊崩掉
        depth_f32 = np.require(d32, dtype=np.float32, requirements=["C", "A", "O"])
        # 給 log 看範圍
        log(sub, f"depth_f32(m) range={float(depth_f32.min()):.3f}~{float(depth_f32.max()):.3f} m | trunc={depth_trunc}")
        log_ok(sub)
    except Exception as e:
        log_fail(sub, e); raise

    # 4) 建立 Open3D 的 RGBD（float32 m + depth_scale=1.0）
    sub = "4.make_rgbd(f32m)"
    try:
        # color 也用強制連續/對齊，然後轉 Image
        color_rgb = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)
        color_rgb = np.require(color_rgb, dtype=np.uint8, requirements=["C", "A", "O"])
        color_o3d = o3d.geometry.Image(color_rgb.copy())

        depth_o3d = o3d.geometry.Image(depth_f32)  # float32(m)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0,                # ← 重點：單位是「公尺」
            depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False,
        )
        log_ok(sub)
    except Exception as e:
        log_fail(sub, e); raise

    # 5) 建立點雲（顯式給 extrinsic = I，避開某些 overloading 小雷）
    sub = "5.create_pcd"
    try:
        extrinsic = np.eye(4, dtype=np.float64)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsic, extrinsic)
        n0 = np.asarray(pcd.points).shape[0]
        log(sub, f"points(after create)={n0}")
        log_ok(sub)
    except Exception:
        log_exc(sub); raise


    # 6) 降採樣
    sub = "6.voxel_down_sample"
    try:
        if voxel_size and voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        n1 = np.asarray(pcd.points).shape[0]
        log(sub, f"points(after voxel)={n1}")
        log_ok(sub)
    except Exception:
        log_exc(sub)
        raise

    # 7) 去雜訊
    sub = "7.remove_statistical_outlier"
    try:
        if nb_neighbors and std_ratio and nb_neighbors > 0 and std_ratio > 0:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=int(nb_neighbors),
                std_ratio=float(std_ratio),
            )
        n2 = np.asarray(pcd.points).shape[0]
        log(sub, f"kept={n2} points")
        log_ok(sub)
    except Exception:
        log_exc(sub)
        raise

    # 8) ROI 裁切（世界座標）
    sub = "8.crop_roi"
    try:
        if ROI is not None:
            aabb = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=(ROI["x_min"], -1.0, ROI["z_min"]),
                max_bound=(ROI["x_max"],  1.0, ROI["z_max"]),
            )
            pcd = pcd.crop(aabb)
            n3 = np.asarray(pcd.points).shape[0]
            log(sub, f"ROI=({ROI['x_min']},{ROI['x_max']})x({ROI['z_min']},{ROI['z_max']}); points(after crop)={n3}")
        log_ok(sub)
    except Exception:
        log_exc(sub)
        raise

    # 9) 結束
    sub = "9.finish"
    log(sub, f"FINAL points={np.asarray(pcd.points).shape[0]}")
    log_ok(step)
    return pcd


def pcd_to_occupancy_from_o3d(pcd, draw_annot=False,
                              *, halfcircle_radius_m=0.5, robot_radius_m=0.18,
                              dilate_iters=1):
    """
    將 Open3D 點雲投影到 2D 佔據格：
      - grid_raw: 僅投影到格子（1=有點/疑似障礙）
      - grid_block: 經半圓kernel膨脹後（1=不可行）
      - img_color: 可視化（白=可行, 黑=障礙）
    """
    step = "pcd_to_occupancy_from_o3d"
    try:
        # 0) 基本檢查
        if pcd is None or len(pcd.points) == 0:
            grid_raw = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            img_color = np.dstack([grid_raw * 0 + 255] * 3)
            log(step, "空點雲")
            return grid_raw, grid_raw, img_color

        # 1) 取點，過濾異常數值（避免後續運算出界或崩）
        pts = np.asarray(pcd.points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"pcd.points 形狀不合法: {pts.shape}")
        # 去 NaN/Inf
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        if pts.size == 0:
            grid_raw = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            img_color = np.dstack([grid_raw * 0 + 255] * 3)
            log(step, "點雲全部為非有限值，輸出空佔據格")
            return grid_raw, grid_raw, img_color

        # 2) 準備佔據格與 ROI 基本量
        grid_raw = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        x_min, x_max = ROI["x_min"], ROI["x_max"]
        z_min, z_max = ROI["z_min"], ROI["z_max"]
        # 避免除以0
        assert x_max > x_min and z_max > z_min, "ROI 無效：x/z 範圍需為正"
        x_res = (x_max - x_min) / GRID_SIZE
        z_res = (z_max - z_min) / GRID_SIZE

        # 3) 3D → 2D 投影到格子（x→col, z→row；z 越大越靠前 → row 越小）
        gx = ((pts[:, 0] - x_min) / x_res)
        gz = ((z_max - pts[:, 2]) / z_res)
        # 只收在格內的點（使用 floor 更直覺）
        gx = np.floor(gx).astype(np.int32)
        gz = np.floor(gz).astype(np.int32)
        in_mask = (gx >= 0) & (gx < GRID_SIZE) & (gz >= 0) & (gz < GRID_SIZE)
        if in_mask.any():
            grid_raw[gz[in_mask], gx[in_mask]] = 1

        # 4) 產半圓 kernel（機器人在最前端；半徑預設 0.5 m）
        #    這裡使用你改良過的版本：lock_robot_at_tip=True
        K, anchor = make_forward_semicircle_kernel(
            x_res, z_res,
            robot_radius_m=robot_radius_m,
            halfcircle_radius_m=halfcircle_radius_m,
            lock_robot_at_tip=True
        )
        # OpenCV 需求：kernel 與輸入都用 uint8
        K = K.astype(np.uint8, copy=False)
        grid_in = grid_raw.astype(np.uint8, copy=False)

        # 5) 膨脹（把障礙擴張成不可行區域）
        iters = max(1, int(dilate_iters))
        grid_block = cv2.dilate(grid_in, K, anchor=anchor, iterations=iters)
        # 二值化保險，確保只有 0/1
        grid_block = (grid_block > 0).astype(np.uint8)

        # 6) 可視化（白=可行, 黑=障礙）
        img_gray = (1 - grid_block) * 255
        img_color = cv2.cvtColor(img_gray.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)

        # 7) 格線（可選）
        if draw_annot:
            stepn = max(1, GRID_SIZE // 8)  # 自動分 8 段
            for i in range(0, GRID_SIZE, stepn):
                cv2.line(img_color, (i, 0), (i, GRID_SIZE - 1), (180, 180, 180), 1)
                cv2.line(img_color, (0, i), (GRID_SIZE - 1, i), (180, 180, 180), 1)

        log(step, f"grid_block.sum={int(grid_block.sum())} (1=不可行)")
        return grid_raw, grid_block, img_color

    except Exception:
        log_exc(step)
        raise

# ===== A* (memory-lean, precise, 8-dir with octile / chebyshev heuristic) =====
import heapq
import numpy as np
import cv2

# 方向與對應成本（前 4 個直走、後 4 個對角）
_DIRS8 = np.array([[-1,  0],  # 0 up
                   [ 1,  0],  # 1 down
                   [ 0, -1],  # 2 left
                   [ 0,  1],  # 3 right
                   [-1, -1],  # 4 up-left
                   [-1,  1],  # 5 up-right
                   [ 1, -1],  # 6 down-left
                   [ 1,  1]], # 7 down-right
                  dtype=np.int8)

_DIRS4 = _DIRS8[:4]

def _h_octile(a, b, diag_cost=1.41421356237):
    """8向 + 直走=1、對角=diag_cost 的 admissible/consistent 啟發式"""
    dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1])
    D, D2 = 1.0, float(diag_cost)
    return (D * (dr + dc) + (D2 - 2 * D) * min(dr, dc))

def _h_manhattan(a, b):
    """4向 + 單步=1 的啟發式"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_opt(
    grid_free: np.ndarray,              # 2D, 1/True=可走, 0/False=障礙
    start: tuple, goal: tuple,          # (row, col)
    *,
    allow_diagonal: bool = True,
    diag_cost: float = 1.41421356237,   # √2；若想等權重就傳 1.0
    avoid_corner_cut: bool = True
):
    """
    回傳: path = [(r,c), ...]；若無解回傳 []
    記憶體配置：g(float32), parent_dir(int8), closed(bool) → 都是 HxW。
    """
    step = "astar_opt"
    H, W = grid_free.shape
    sr, sc = map(int, start); gr, gc = map(int, goal)

    # 邊界/可行性
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        log(step, "❌ start/goal 超出邊界"); return []
    if grid_free[sr, sc] == 0 or grid_free[gr, gc] == 0:
        log(step, "❌ 起點或終點在不可行區"); return []
    if (sr, sc) == (gr, gc):
        return [(sr, sc)]

    # 選方向、成本、啟發式
    if allow_diagonal:
        dirs = _DIRS8
        # 各方向步長成本（直=1，對角=diag_cost）
        step_cost = np.ones(8, dtype=np.float32)
        step_cost[4:] = float(diag_cost)
        heur = (lambda p, q: _h_octile(p, q, diag_cost))
    else:
        dirs = _DIRS4
        step_cost = np.ones(4, dtype=np.float32)
        heur = _h_manhattan

    # 配陣列（省記憶體）
    g = np.full((H, W), np.inf, dtype=np.float32)
    parent_dir = np.full((H, W), -1, dtype=np.int8)   # 存方向索引 0..7
    closed = np.zeros((H, W), dtype=np.bool_)

    g[sr, sc] = 0.0
    h0 = heur((sr, sc), (gr, gc))
    # heap: (f, h, r, c) ；以 h 做 tie-break，傾向往目標前進
    open_heap = [(h0, h0, sr, sc)]
    heapq.heapify(open_heap)

    while open_heap:
        f, h, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        if (r, c) == (gr, gc):
            # 回溯（依 parent_dir 走回去）
            path = []
            rr, cc = r, c
            while not (rr == sr and cc == sc):
                path.append((rr, cc))
                k = parent_dir[rr, cc]
                if k < 0:   # 無法回溯，代表資料壞了或競態
                    return []
                dr, dc = dirs[k]
                rr -= dr; cc -= dc
            path.append((sr, sc))
            path.reverse()
            return path

        closed[r, c] = True
        base_g = g[r, c]

        for k in range(len(dirs)):
            dr = int(dirs[k, 0]); dc = int(dirs[k, 1])
            r2, c2 = r + dr, c + dc
            if not (0 <= r2 < H and 0 <= c2 < W):
                continue
            if grid_free[r2, c2] == 0:
                continue

            # 避免「鑽牆角」
            if allow_diagonal and avoid_corner_cut and dr != 0 and dc != 0:
                if grid_free[r, c + dc] == 0 and grid_free[r + dr, c] == 0:
                    continue

            # 單步成本（直=1；對角=diag_cost）
            g2 = base_g + step_cost[k]
            if g2 < g[r2, c2]:
                g[r2, c2] = g2
                parent_dir[r2, c2] = k
                hh = heur((r2, c2), (gr, gc))
                ff = g2 + hh
                heapq.heappush(open_heap, (ff, hh, r2, c2))

    log(step, "⚠️ 無可行路徑")
    return []


def draw_and_save_path(grid_block, vis_img_bgr, start_rc, goal_rc,
                       out_png=None, out_csv=None,
                       *, allow_diagonal=True, diag_cost=1.41421356237, avoid_corner_cut=True):
    """
    grid_block: 1=不可行, 0=可走
    """
    step = "draw_and_save_path"
    try:
        H, W = grid_block.shape
        grid_free = (grid_block == 0).astype(np.uint8)

        sr, sc = start_rc; gr, gc = goal_rc
        if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
            log(step, "❌ start/goal 超出邊界"); return vis_img_bgr
        if grid_free[sr, sc] == 0:
            log(step, "⚠️ 起點在不可行區"); return vis_img_bgr
        if grid_free[gr, gc] == 0:
            log(step, "⚠️ 終點在不可行區"); return vis_img_bgr

        path = astar_opt(grid_free, start_rc, goal_rc,
                         allow_diagonal=allow_diagonal,
                         diag_cost=diag_cost,
                         avoid_corner_cut=avoid_corner_cut)

        img = vis_img_bgr.copy()
        if len(path) > 0:
            for r, c in path:
                cv2.circle(img, (c, r), 1, (0, 0, 255), -1)
            cv2.circle(img, (sc, sr), 4, (0, 255, 255), -1)  # start 黃
            cv2.circle(img, (gc, gr), 4, (0, 255,   0), -1)  # goal 綠

            if out_png:
                cv2.imwrite(out_png, img)
                log(step, f"存檔 {out_png}，路徑點數={len(path)}")

            if out_csv:
                with open(out_csv, "w", encoding="utf-8") as f:
                    f.write("x_m,z_m\n")
                    for r, c in path:
                        x, z = cell_to_meters(r, c)
                        f.write(f"{x:.6f},{z:.6f}\n")
                log(step, f"存檔 {out_csv}")
        else:
            log(step, "⚠️ 找不到路徑，僅輸出底圖")
            if out_png:
                cv2.imwrite(out_png, img)

        return img
    except Exception:
        log_exc(step); raise

def main():
    # --- 啟動與環境 ---
    debug_env()
    log("main", "✅ RealSense 初始化中…")
    pipeline, profile, align, rs_scale = auto_start_realsense()
    log("main", "✅ 串流開始：C=拍照(桌面)｜P=跑A*｜Q=離開")

    # A* 起終點（meters_to_cell 需是只回 (row,col) 的版本）
    start_rc = meters_to_cell(*A_STAR_START_WORLD)
    goal_rc  = meters_to_cell(*A_STAR_GOAL_WORLD)
    log("main", f"A*: start_rc={start_rc} goal_rc={goal_rc}")

    last_update = 0.0
    tick = time.monotonic()

    occ_block = np.zeros((GRID_SIZE, GRID_SIZE), np.uint8)
    occ_vis   = np.dstack([occ_block*0 + 255]*3)

    pool = ThreadPoolExecutor(max_workers=2)

    try:
        # 先建一次 O3D intrinsic（避免每幀重建）
        log("main", "建立 Open3D Intrinsic")
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        o3d_intrinsic = build_o3d_intrinsic_from_frame(color_frame)

        # ✍️ 善意提醒：若 rs_scale ≠ 0.001，而你目前的 frames_to_pointcloud 是「mm+1000」路徑，
        # 請改用我提供的「float32(m)+depth_scale=1.0」版本，或在函式內加 depth_format 開關。
        if abs(rs_scale - 0.001) > 1e-6:
            log("main", f"⚠️ 注意：rs_scale={rs_scale:.6f}（非 0.001）。"
                        " 目前 pointcloud 流程假設深度為 mm。建議切到 float32(m) 安全路徑。")

        while True:
            step_loop = "main_loop_fetch"
            try:
                frames = pipeline.wait_for_frames()
                frames = align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    log(step_loop, "缺幀，略過")
                    continue

                # ===== 重要：拷貝成 numpy（避免 frame 壽命問題） =====
                depth_np = np.asanyarray(depth_frame.get_data()).copy()
                color_np = np.asanyarray(color_frame.get_data()).copy()
                if depth_np.size == 0 or color_np.size == 0:
                    log(step_loop, "空影像，略過")
                    continue

                # 週期性更新地圖
                now = time.monotonic()
                if (now - tick) >= MAP_UPDATE_INTERVAL_SEC:
                    tick = now

                    # 1) 顯示 Color/Depth
                    try:
                        cv2.imshow("Color", color_np)
                        cv2.imshow("Depth", colorize_depth(depth_np))
                    except Exception:
                        log_exc("imshow Color/Depth")

                    # 2) Open3D：RGBD → PCD
                    # 合理截斷距離（公尺），以 ROI 前/後半徑為上限（可再加總上限，如 10m）
                    depth_trunc_m = max(
                        0.1,
                        min(10.0, float(ROI["z_max"]) if ROI["z_max"] > 0 else abs(float(ROI["z_min"])))
                    )

                    pcd = frames_to_pointcloud_o3d(
                        color_np=color_np,
                        depth_np=depth_np,
                        o3d_intrinsic=o3d_intrinsic,
                        depth_trunc=depth_trunc_m,   # 公尺
                        ROI=None,                    # 此處先不裁，下一步才做
                    )

                    # 3) PCD → 佔據格
                    _, occ_block, occ_vis = pcd_to_occupancy_from_o3d(
                        pcd, draw_annot=DRAW_GRID_ANNOTATION
                    )

                    # 4) 輸出/顯示
                    try:
                        np.save("occupancy_map.npy", occ_block)
                        cv2.imwrite("occupancy_map_annotated.png", occ_vis)
                        cv2.imshow("Occupancy (live)", occ_vis)
                        log("main_loop_render", f"地圖更新：occ_block.shape={occ_block.shape}, sum(block)={int(occ_block.sum())}")
                    except Exception:
                        log_exc("save/show occupancy")

                # 鍵盤事件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # 背景存三檔（Open3D 版；只傳 numpy）
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    color_path = os.path.join(DESKTOP, f"color_{ts}.png")
                    depth_path = os.path.join(DESKTOP, f"depth_{ts}.png")
                    ply_path   = os.path.join(DESKTOP, f"pointcloud_{ts}.ply")

                    color_copy = color_np.copy()
                    depth_copy = depth_np.copy()          # 通常是 uint16 (裝置單位)
                    o3d_intr_copy = o3d_intrinsic
                    depth_trunc_copy = depth_trunc_m

                    def save_triplet_o3d():
                        step_bg = "save_triplet_o3d"
                        try:
                            log(step_bg, "寫入 color/depth 影像")
                            cv2.imwrite(color_path, color_copy)
                            cv2.imwrite(depth_path, depth_copy)

                            log(step_bg, "numpy→O3D RGBD")
                            color_rgb = cv2.cvtColor(color_copy, cv2.COLOR_BGR2RGB)
                            color_o3d = o3d.geometry.Image(color_rgb.copy())

                            # RealSense: depth_in_meter = depth_value * rs_scale
                            # Open3D: depth_in_meter = depth_value / depth_scale
                            # 故 depth_scale = 1 / rs_scale
                            depth_u16 = np.ascontiguousarray(depth_copy.astype(np.uint16))
                            depth_o3d = o3d.geometry.Image(depth_u16)

                            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                color_o3d, depth_o3d,
                                depth_scale=(1.0 / float(rs_scale)),
                                depth_trunc=float(depth_trunc_copy),
                                convert_rgb_to_intensity=False
                            )

                            log(step_bg, "RGBD→PCD")
                            pcd_cap = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intr_copy)
                            pcd_cap = pcd_cap.voxel_down_sample(0.02)
                            pcd_cap, _ = pcd_cap.remove_statistical_outlier(20, 2.0)

                            log(step_bg, f"寫入 PLY {ply_path}，點數={np.asarray(pcd_cap.points).shape[0]}")
                            o3d.io.write_point_cloud(ply_path, pcd_cap)
                            log(step_bg, "✅ 完成")
                        except Exception:
                            log_exc(step_bg)

                    pool.submit(save_triplet_o3d)

                elif key == ord('p'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_png = os.path.join(DESKTOP, f"astar_path_{ts}.png")
                    out_csv = os.path.join(DESKTOP, f"astar_path_{ts}.csv")
                    _ = draw_and_save_path(occ_block, occ_vis, start_rc, goal_rc, out_png, out_csv)

                elif key == ord('q'):
                    break

            except Exception:
                log_exc(step_loop)
                # 不中斷整個程式，繼續 loop
                continue

    finally:
        try:
            pool.shutdown(wait=True)  # 等待背景存檔完成再退出
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        log("main", "✅ 已結束")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

