import os
import csv
from datetime import datetime
import torch

class TactileDataLogger:
    """一个专门用于记录触觉力数据到CSV文件的类。"""

    def __init__(self, task_name: str, log_dir: str):
        """
        初始化Logger。

        Args:
            task_name: 当前任务的名称，用于文件名。
            log_dir: 存放CSV文件的目录。
        """
        self.task_name = task_name
        self.log_dir = log_dir
        self.csv_file = None
        self.csv_writer = None
        self.episode_count = 0

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"[TactileDataLogger] Logging CSV data to: {self.log_dir}")

    def start_new_episode(self):
        """关闭旧文件（如果存在），并为新回合创建一个新的CSV文件。"""
        self.close()  # 确保上一个文件被正确关闭
        self.episode_count += 1
        
        # 构造文件名
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        filename = f"{self.task_name}_ep{self.episode_count:03d}_{timestamp}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        # 创建并打开文件，写入表头
        self.csv_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'is_engaged', 'is_engaged_half', 'is_success', 'normal_sum', 'shear_x_sum', 'shear_y_sum'])
        print(f"[TactileDataLogger] Started new log file for Episode {self.episode_count}: {filename}")

    def log_step(self, step: int, is_engaged: bool, is_engaged_half: bool, is_success: bool,tactile_force: torch.Tensor):
        """
        在每个仿真步记录一行数据。

        Args:
            step: 当前的仿真步数。
            is_engaged: 当前是否满足 engaged 条件。
            tactile_force: 3维的触觉力张量。
        """
        if self.csv_writer is None:
            return # 如果文件未打开，则不记录

        # 将数据转换为可写入的格式
        # .item() 用于从0维张量中提取Python数字
        # is_engaged 是布尔值，转为 0/1
        row = [
            step,
            1 if is_engaged else 0,
            1 if is_engaged_half else 0,
            1 if is_success else 0,
            tactile_force[0].item(),
            tactile_force[1].item(),
            tactile_force[2].item()
        ]
        self.csv_writer.writerow(row)

    def close(self):
        """关闭当前打开的CSV文件。"""
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[TactileDataLogger] Closed log file for Episode {self.episode_count - 1}.")
            self.csv_file = None
            self.csv_writer = None