#!/usr/bin/env python3
import os
import json
import numpy as np
import argparse
from tqdm import tqdm

# 可选：使用 imageio（无 GUI 依赖，推荐）
try:
    import imageio.v3 as iio
    USE_IMAGEIO = True
except ImportError:
    import cv2
    USE_IMAGEIO = False

# 可视化依赖
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def visualize_force_field_arrows(force_field_np, downsample=8, figsize=(2.24, 2.24), dpi=100):
    """
    将 [3, 224, 224] 的力场转为箭头图（28x28）
    返回 PIL Image
    """
    # force_field_np: [3, H, W] → [H, W, 3]
    force_field = np.transpose(force_field_np, (1, 2, 0))  # [224,224,3]
    H, W = force_field.shape[:2]

    step = downsample
    x = np.arange(0, W, step)
    y = np.arange(0, H, step)
    X, Y = np.meshgrid(x, y)

    fx = force_field[Y, X, 0]
    fy = force_field[Y, X, 1]
    # fz = force_field[Y, X, 2]  # 可选：用于颜色

    # 箭头长度归一化（可选）
    magnitude = np.sqrt(fx**2 + fy**2)
    max_mag = magnitude.max()
    if max_mag > 0:
        fx = fx / max_mag
        fy = fy / max_mag

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.quiver(X, Y, fx, fy, scale=20, width=0.005, headwidth=3, headlength=5)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 转 PIL
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    pil_img = Image.fromarray(buf)
    plt.close(fig)
    return pil_img

def make_episode_video(episode_dir, fps=30, downsample=8):
    meta_file = os.path.join(episode_dir, "episode_info.json")
    if not os.path.exists(meta_file):
        print(f"[Error] No episode_info.json in {episode_dir}")
        return

    with open(meta_file, 'r') as f:
        meta = json.load(f)

    success = meta.get("success", False)
    success_step = meta.get("success_step", -1)
    total_steps = meta["total_steps"]
    interval = meta.get("record_interval", 1)

    # 获取所有 frame 索引
    frame_indices = []
    for i in range(0, total_steps + 1, interval):
        ff_path = os.path.join(episode_dir, "force_field", f"force_field_{i:06d}.npy")
        cam_path = os.path.join(episode_dir, "camera", f"frame_{i:06d}.png")
        if os.path.exists(ff_path) and os.path.exists(cam_path):
            frame_indices.append(i)

    if not frame_indices:
        print(f"[Warning] No matching frames in {episode_dir}")
        return

    # 读取一帧确定尺寸
    sample_cam = np.array(Image.open(os.path.join(episode_dir, "camera", f"frame_{frame_indices[0]:06d}.png")))
    cam_h, cam_w = sample_cam.shape[:2]

    # 箭头图尺寸 = 224x224（与原始力场一致）
    ff_w, ff_h = 224, 224

    out_w = ff_w + cam_w
    out_h = max(ff_h, cam_h)

    output_path = os.path.join(episode_dir, "tactile_camera_video.mp4")

    if USE_IMAGEIO:
        writer = iio.imopen(output_path, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=fps)
        frames_to_write = []
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for step in tqdm(frame_indices, desc=f"Processing {os.path.basename(episode_dir)}"):
        # 1. 读取力场并可视化
        ff_data = np.load(os.path.join(episode_dir, "force_field", f"force_field_{step:06d}.npy"))  # [3,224,224]
        ff_img = visualize_force_field_arrows(ff_data, downsample=downsample)  # PIL Image 224x224

        # 2. 读取相机图像
        cam_img = Image.open(os.path.join(episode_dir, "camera", f"frame_{step:06d}.png")).convert('RGB')
        if cam_img.size != (cam_w, cam_h):
            cam_img = cam_img.resize((cam_w, cam_h))

        # 3. 拼接
        combined = Image.new('RGB', (out_w, out_h), (0, 0, 0))
        combined.paste(ff_img, (0, 0))
        combined.paste(cam_img, (ff_w, (out_h - cam_h) // 2))

        # 4. 转 numpy
        combined_np = np.array(combined)

        if USE_IMAGEIO:
            frames_to_write.append(combined_np)
        else:
            if combined_np.shape[1] != out_w or combined_np.shape[0] != out_h:
                combined_np = cv2.resize(combined_np, (out_w, out_h))
            video_writer.write(cv2.cvtColor(combined_np, cv2.COLOR_RGB2BGR))

    if USE_IMAGEIO:
        writer.write(frames_to_write)
        writer.close()
    else:
        video_writer.release()

    print(f"[Success] Video saved to {output_path}")
    print(f"    → Success: {success}, Success Step: {success_step}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tactile-camera video from episode data")
    parser.add_argument("--episode_dir", type=str, required=True, help="Path to episode directory")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--downsample", type=int, default=8, help="Downsample factor for quiver (8 → 28x28)")
    args = parser.parse_args()

    make_episode_video(
        episode_dir=args.episode_dir,
        fps=args.fps,
        downsample=args.downsample
    )