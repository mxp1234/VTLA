#!/usr/bin/env python3
"""
多视角触觉-视觉数据可视化脚本
--------------------------------
该脚本将episode数据目录中的六种不同数据源拼接成一个3x2网格视频：
1. 左上：三维力场可视化（箭头图）
2. 右上：相机视频（环境视角）
3. 左中：左GelSight传感器RGB图像
4. 右中：右GelSight传感器RGB图像
5. 左下：左GelSight传感器Marker Motion可视化
6. 右下：右GelSight传感器Marker Motion可视化


用法：
python tactile_recording_vis.py \
    --episode_dir ./tactile_recordings/episode_0000_L_hole_fail_wo_tactile_400_epoch_w0_obs-noise \
    --video_path /home/pi-zero/isaac-sim/TacEx/logs/rl_games/Peg-in-hole/Lhole_III/videos/play/L_hole_fail_wo_tactile_400_epoch_w0_obs-noise.mp4 \
    --output_path ./output/episode_0000_L_hole_fail_wo_tactile_400_epoch_w0_obs-noise.mp4 \
    --fps 30 \
    --downsample 8
"""

import argparse
import json
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate combined visualization from tactile data.')
    parser.add_argument('--episode_dir', type=str, required=True,
                        help='Path to episode directory containing data')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to the recorded video file')
    parser.add_argument('--output_path', type=str, default='./combined_video.mp4',
                        help='Output video file path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    parser.add_argument('--downsample', type=int, default=8,
                        help='Downsample factor for force field quiver plot (e.g., 8 → 28x28)') 
    return parser.parse_args()

def create_force_field_visualization(force_field_data):
    """从原始力场数据创建可视化图像（颜色根据当前数据自动归一化）"""
    try:
        X = np.array(force_field_data['grid_x'])
        Y = np.array(force_field_data['grid_y'])
        U = np.array(force_field_data['shear_x'])   # X方向切向力
        V = np.array(force_field_data['shear_y'])   # Y方向切向力
        C = np.array(force_field_data['normal_z'])  # 法向力
    except KeyError as e:
        print(f"Missing force field data key: {e}")
        return None

    vis_size = 224
    vis_img = np.ones((vis_size, vis_size, 3), dtype=np.uint8) * 255  # 白色背景

    if X.shape != (28, 28) or Y.shape != (28, 28):
        print(f"Unexpected force field grid shape: X={X.shape}, Y={Y.shape}")
        return None

    # === ✅ 自动归一化法向力范围 ===
    min_normal = np.min(C)
    max_normal = np.max(C)
    if np.isclose(max_normal, min_normal):
        max_normal = min_normal + 1e-6  # 防止除0
    norm_c = (C - min_normal) / (max_normal - min_normal)

    # === 可视化 ===
    for i in range(28):
        for j in range(28):
            # 坐标
            x = int(X[i, j])
            y = int(Y[i, j])

            # 切向力缩放
            u = U[i, j] * 5
            v = V[i, j] * 5
            c_val = C[i, j]

            # 末端坐标
            end_x = int(np.clip(x + u, 0, vis_size - 1))
            end_y = int(np.clip(y + v, 0, vis_size - 1))

            # === 颜色映射：蓝→红渐变 ===
            color_val = norm_c[i, j]
            color = (
                int(255 * color_val),       # 红通道
                0,
                int(255 * (1 - color_val))  # 蓝通道
            )

            # 绘制箭头
            cv2.arrowedLine(vis_img, (x, y), (end_x, end_y), color, 1, tipLength=0.1)

            # 圆点大小按法向力幅值变化
            normal_mag = abs(c_val)
            normal_mag = np.clip(normal_mag * 5, 1, 8)
            cv2.circle(vis_img, (x, y), int(normal_mag), color, -1)

    # === 标注 ===
    # cv2.putText(vis_img, "Force Field Visualization", (10, 25),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    # cv2.putText(vis_img, f"Normal range: [{min_normal:.2f}, {max_normal:.2f}]", (10, vis_size - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return vis_img


def load_force_field_data(episode_dir, target_timestamp):
    """
    加载并可视化力场原始数据（修复文件名匹配和数据加载问题）
    """
    force_field_dir = os.path.join(episode_dir, "force_field")
    
    # 获取所有可用的时间戳
    timestamps = []
    if not os.path.exists(force_field_dir):
        print(f"[Warning] Force field directory not found: {force_field_dir}")
        return None
        
    for f in sorted(os.listdir(force_field_dir)):
        if f.startswith("data_") and f.endswith(".json"):
            try:
                # 从文件名提取时间戳 (e.g., data_000123.json -> 123)
                timestamp = int(f.split('_')[1].split('.')[0])
                timestamps.append(timestamp)
            except (ValueError, IndexError):
                continue
    
    if not timestamps:
        print("[Warning] No force field data files found")
        return None
    
    # 找到最接近目标时间戳的文件
    closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))
    
    # 读取数据
    file_path = os.path.join(force_field_dir, f"data_{closest_timestamp:06d}.json")
    
    if not os.path.exists(file_path):
        print(f"[Warning] Force field file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            force_field_data = json.load(f)
        # 创建可视化
        return create_force_field_visualization(force_field_data)
    except Exception as e:
        print(f"Error loading force field data from {file_path}: {e}")
        return None

def load_marker_motion_data(episode_dir, target_timestamp, side='left',
                            vis_size=224, orig_size=(480, 640)):
    """
    加载 marker motion 数据（像素坐标）并生成可视化（等比缩放到 vis_size x vis_size 并居中）

    """
    marker_dir = os.path.join(episode_dir, "marker_motion")

    # 检查目录是否存在
    if not os.path.exists(marker_dir):
        print(f"[Warning] Marker motion directory not found: {marker_dir}")
        return None

    # 获取所有可用的时间戳
    timestamps = []
    prefix = f"{side}_markers_"
    for fname in sorted(os.listdir(marker_dir)):
        if fname.startswith(prefix) and fname.endswith(".json"):
            try:
                # 从文件名提取时间戳 (e.g., left_markers_000123.json -> 123)
                timestamp = int(fname.split('_')[-1].split('.')[0])
                timestamps.append(timestamp)
            except (ValueError, IndexError):
                continue

    if not timestamps:
        print(f"[Warning] No {side} marker motion files found")
        return None

    # 找到最接近目标时间戳的文件
    closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))

    # 读取数据
    file_path = os.path.join(marker_dir, f"{side}_markers_{closest_timestamp:06d}.json")
    if not os.path.exists(file_path):
        print(f"[Warning] Marker motion file not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 验证数据结构
        if 'data' not in data or 'initial_positions' not in data['data'] or 'current_positions' not in data['data']:
            print(f"[Warning] Invalid marker motion data structure in {file_path}")
            return None

        initial_positions = np.array(data['data']['initial_positions'])
        current_positions = np.array(data['data']['current_positions'])

        # 验证数据形状
        if initial_positions.ndim != 2 or current_positions.ndim != 2 or initial_positions.shape[1] != 2 or current_positions.shape[1] != 2:
            print(f"[Warning] Invalid marker data shape in {file_path}: "
                  f"initial={initial_positions.shape}, current={current_positions.shape}")
            return None

        # 准备等比缩放参数：orig_size 是 (height, width)
        orig_h, orig_w = orig_size
        target = vis_size
        # 使用等比缩放 (fit into square) 并居中填充
        scale = min(target / orig_w, target / orig_h)  # 注意以宽高分别比较
        pad_x = (target - orig_w * scale) / 2.0
        pad_y = (target - orig_h * scale) / 2.0

        # 创建白色背景画布
        vis_img = np.ones((target, target, 3), dtype=np.uint8) * 255

        # 限制处理的标记点数量（最多99个）
        num_markers = min(len(initial_positions), len(current_positions), 99)

        # 绘制所有标记点和位移箭头（将原始像素坐标映射到 vis_size 上）
        for i in range(num_markers):
            # 假设 initial_positions[:,0] = x (列, width), [:,1] = y (行, height)
            ox, oy = initial_positions[i, 0], initial_positions[i, 1]
            cx, cy = current_positions[i, 0], current_positions[i, 1]

            # 映射并四舍五入为整数像素坐标，同时做边界裁切
            sx = int(np.clip(round(ox * scale + pad_x), 0, target - 1))
            sy = int(np.clip(round(oy * scale + pad_y), 0, target - 1))
            ex = int(np.clip(round(cx * scale + pad_x), 0, target - 1))
            ey = int(np.clip(round(cy * scale + pad_y), 0, target - 1))

            start = (sx, sy)
            end = (ex, ey)

            # 绘制初始marker点（红色），当前marker点（蓝色），以及位移箭头（黑色）
            cv2.circle(vis_img, start, 2, (0, 0, 255), -1)
            cv2.circle(vis_img, end, 2, (255, 0, 0), -1)
            # 当 start==end 时 arrowedLine 有时不绘制，防护一下
            if start != end:
                cv2.arrowedLine(vis_img, start, end, (0, 0, 0), 1, tipLength=0.1)

        # 添加标题（字体大小随缩放略作调整）
        # scale_font 用于在原始 224 基准上缩放文字大小（如果原始尺寸不同）
        scale_font = max(0.4, 0.6 * scale)
        # cv2.putText(vis_img, f"{side.title()} Marker Motion", (10, int(25 * max(1.0, scale))),
        #             cv2.FONT_HERSHEY_SIMPLEX, scale_font, (0, 0, 0), 1)
        # cv2.putText(vis_img, f"Markers: {num_markers}", (10, target - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, max(0.3, 0.4 * scale_font), (0, 0, 0), 1)

        return vis_img

    except Exception as e:
        print(f"Error loading marker motion data from {file_path}: {e}")
        return None


def get_file_timestamps(directory, prefix, file_ext):
    """获取指定目录中按命名规则排列的文件时间戳列表（修复时间戳提取逻辑）"""
    timestamps = []
    if not os.path.exists(directory):
        return timestamps
        
    for f in sorted(os.listdir(directory)):
        if f.startswith(prefix) and f.endswith(file_ext):
            try:
                # 从文件名提取时间戳 (e.g., left_000123.png -> 123)
                # 处理不同命名格式
                parts = f.split('_')
                if len(parts) < 2:
                    continue
                timestamp_str = parts[-1].split('.')[0]
                timestamp = int(timestamp_str)
                timestamps.append(timestamp)
            except (ValueError, IndexError):
                continue
    return sorted(timestamps)

def load_tactile_rgb(episode_dir, target_timestamp, side='left'):
    """
    加载触觉传感器RGB图像，使用最近邻匹配
    """
    rgb_dir = os.path.join(episode_dir, "rgb")
    if not os.path.exists(rgb_dir):
        print(f"[Warning] RGB directory not found: {rgb_dir}")
        return None
    
    timestamps = get_file_timestamps(rgb_dir, f"{side}_", ".png")
    
    if not timestamps:
        print(f"[Warning] No {side} RGB files found")
        return None
    
    # 找到最近的时间戳
    closest_timestamp = min(timestamps, key=lambda x: abs(x - target_timestamp))
    
    img_path = os.path.join(rgb_dir, f"{side}_{closest_timestamp:06d}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Warning] Failed to read image: {img_path}")
        return img
    else:
        print(f"[Warning] RGB file not found: {img_path}")
        return None

def load_camera_frame(video_path, frame_idx):
    """
    从指定视频文件中加载特定帧（修复视频读取问题）
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return None
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("[ERROR] Video has zero frames")
        cap.release()
        return None
    
    # 处理越界帧索引
    if frame_idx >= total_frames:
        print(f"[WARNING] Requested frame index {frame_idx} exceeds video length ({total_frames} frames). Using last frame.")
        frame_idx = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[ERROR] Failed to read frame at index {frame_idx}")
        return None
    
    return frame

def create_placeholder_image(text, width=480, height=240):
    """创建占位图像（当数据缺失时使用）"""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128  # 灰色背景
    font_scale = min(1.0, width/300)
    thickness = max(1, int(width/200))
    
    # 计算文本位置
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return img

def create_combined_frame(episode_dir, video_path, frame_idx, sim_total_steps):
    """
    创建3x2网格拼接帧（修复数据对齐和缺失处理）
    """
    # 计算当前仿真时间戳（假设视频帧与仿真步数线性对应）
    # video_total_frames = 150 (固定)
    video_total_frames = 150
    timestamp = int(frame_idx * sim_total_steps / video_total_frames)
    
    # 1. 加载所有数据
    ff_vis = load_force_field_data(episode_dir, timestamp)
    camera_img = load_camera_frame(video_path, frame_idx)
    tactile_left = load_tactile_rgb(episode_dir, timestamp, side='left')
    tactile_right = load_tactile_rgb(episode_dir, timestamp, side='right')
    marker_left = load_marker_motion_data(episode_dir, timestamp, side='left')
    marker_right = load_marker_motion_data(episode_dir, timestamp, side='right')
    
    # 2. 创建一个空白画布用于拼接 (3x2 布局)
    canvas_height = 720  # 3行 x 240像素
    canvas_width = 960   # 2列 x 480像素
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas.fill(30)  # 深灰色背景
    
    # 3. 定义每个区域的位置 (3行2列)
    quad_width = canvas_width // 2
    quad_height = canvas_height // 3
    
    # 4. 为缺失数据创建占位符
    if ff_vis is None:
        ff_vis = create_placeholder_image("NO FORCE FIELD DATA", quad_width, quad_height)
    if camera_img is None:
        camera_img = create_placeholder_image("NO CAMERA DATA", quad_width, quad_height)
    if tactile_left is None:
        tactile_left = create_placeholder_image("NO LEFT TACTILE RGB", quad_width, quad_height)
    if tactile_right is None:
        tactile_right = create_placeholder_image("NO RIGHT TACTILE RGB", quad_width, quad_height)
    if marker_left is None:
        marker_left = create_placeholder_image("NO LEFT MARKER MOTION", quad_width, quad_height)
    if marker_right is None:
        marker_right = create_placeholder_image("NO RIGHT MARKER MOTION", quad_width, quad_height)
    
    # 5. 缩放和放置不同区域 (按3x2网格排列)
    regions = [
        (ff_vis, 0, 0, quad_width, quad_height),                     # 第1行左：力场
        (camera_img, quad_width, 0, quad_width, quad_height),        # 第1行右：相机
        (tactile_left, 0, quad_height, quad_width, quad_height),     # 第2行左：左触觉RGB
        (tactile_right, quad_width, quad_height, quad_width, quad_height),  # 第2行右：右触觉RGB
        (marker_left, 0, 2*quad_height, quad_width, quad_height),    # 第3行左：左marker motion
        (marker_right, quad_width, 2*quad_height, quad_width, quad_height)  # 第3行右：右marker motion
    ]
    
    for img, x, y, w, h in regions:
        if img is None:
            continue
            
        # 保持纵横比缩放
        img_h, img_w = img.shape[:2]
        if img_h == 0 or img_w == 0:
            continue
            
        aspect_ratio = img_w / img_h
        
        # 计算保持纵横比的缩放
        if w / h > aspect_ratio:
            # 宽度受限制
            new_h = h
            new_w = int(h * aspect_ratio)
        else:
            # 高度受限制
            new_w = w
            new_h = int(w / aspect_ratio)
        
        # 确保尺寸有效
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # 重新缩放图像
        try:
            resized_img = cv2.resize(img, (new_w, new_h))
        except cv2.error as e:
            print(f"Resize error: {e}")
            continue
            
        # 居中放置
        x_offset = x + (w - new_w) // 2
        y_offset = y + (h - new_h) // 2
        
        # 复制到画布
        y_end = min(y_offset + new_h, canvas_height)
        x_end = min(x_offset + new_w, canvas_width)
        canvas[y_offset:y_end, x_offset:x_end] = resized_img[:(y_end-y_offset), :(x_end-x_offset)]
        
        # 添加边框
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (100, 100, 100), 1)
    
    # 6. 添加时间戳和标题
    cv2.putText(canvas, f"Frame: {frame_idx}/{video_total_frames-1} | Sim Step: {timestamp}/{sim_total_steps-1}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    # 7. 添加区域标签
    labels = [
        "Force Field", 
        "Camera View",
        "Left GelSight RGB", 
        "Right GelSight RGB",
        "Left Marker Motion", 
        "Right Marker Motion"
    ]
    positions = [
        (10, quad_height//2), 
        (quad_width+10, quad_height//2),
        (10, quad_height + quad_height//2), 
        (quad_width+10, quad_height + quad_height//2),
        (10, 2*quad_height + quad_height//2), 
        (quad_width+10, 2*quad_height + quad_height//2)
    ]
    
    for label, (x, y) in zip(labels, positions):
        cv2.putText(canvas, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return canvas

def generate_combined_video(episode_dir, video_path, output_path, fps=30):
    """
    生成3x2网格拼接视频（修复元数据加载问题）
    """
    # 1. 确保输出路径有正确的扩展名
    if not output_path.lower().endswith('.mp4'):
        output_path += '.mp4'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 2. 读取episode信息以获取总步数
    episode_info_path = os.path.join(episode_dir, "episode_info.json")
    video_total_frames = 150  # 固定视频帧数
    sim_total_steps = 150  # 默认值
    
    if os.path.exists(episode_info_path):
        try:
            with open(episode_info_path, 'r') as f:
                episode_info = json.load(f)
            sim_total_steps = episode_info.get("total_steps", 150)
            print(f"Loaded episode info: total_steps={sim_total_steps}")
        except Exception as e:
            print(f"Error loading episode info: {e}")
    else:
        print(f"[Warning] Episode info file not found: {episode_info_path}. Using default total steps=150.")
    
    # 确保总步数合理
    if sim_total_steps <= 0:
        sim_total_steps = 150
        print(f"[Warning] Invalid total steps ({sim_total_steps}), using default value 150")
    
    print(f"Using total simulation steps: {sim_total_steps}")

    # 3. 创建一个临时列表来存储所有帧
    frames = []

    # 4. 生成每一帧 (150帧)
    for frame_idx in tqdm(range(video_total_frames), desc="Generating video frames"):
        frame = create_combined_frame(episode_dir, video_path, frame_idx, sim_total_steps)
        frames.append(frame)

    # 5. 使用 imageio 保存视频
    print(f"Saving video to {output_path} using imageio...")
    try:
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            format='mp4',
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=1,
            output_params=['-movflags', '+faststart']
        )
        print(f"Video saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        # 尝试保存为GIF作为备选
        gif_path = output_path.replace('.mp4', '.gif')
        print(f"Trying to save as GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=fps)
        print(f"GIF saved as fallback to {gif_path}")

if __name__ == "__main__":
    args = parse_arguments()
    
    if not os.path.exists(args.episode_dir):
        print(f"Error: Episode directory not found: {args.episode_dir}")
        exit(1)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        exit(1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    generate_combined_video(
        args.episode_dir,
        args.video_path,
        args.output_path,
        args.fps
    )