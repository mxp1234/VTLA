"""
查看所有已注册的 IsaacLab 任务
必须先启动 Isaac Sim AppLauncher
"""

# 第一步: 启动 Isaac Sim (必须在导入 isaaclab 之前)
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)  # headless=True 不显示GUI
simulation_app = app_launcher.app

# 第二步: 导入其他模块
import gymnasium as gym
import isaaclab_tasks  # 触发自动注册

print("=" * 80)
print("IsaacLab 所有已注册的任务")
print("=" * 80)
print()

# 获取所有已注册的任务
all_tasks = []
for task_spec in gym.registry.values():
    if "Isaac" in task_spec.id:
        all_tasks.append(task_spec.id)

all_tasks.sort()

# 分类统计
direct_tasks = [t for t in all_tasks if "Direct" in t]
manager_tasks = [t for t in all_tasks if "Direct" not in t]

print(f"总任务数: {len(all_tasks)}")
print(f"  - Direct API: {len(direct_tasks)} 个")
print(f"  - Manager-Based API: {len(manager_tasks)} 个")
print()

# 显示前10个任务
print("前 10 个任务:")
print("-" * 80)
for i, task in enumerate(all_tasks[:], 1):
    print(f"  {i}. {task}")
print()

# 显示如何查看特定任务的详细信息
print("示例: 查看特定任务的详细信息")
print("-" * 80)
example_task = "Isaac-Factory-PegInsert-Direct-v0"
spec = gym.spec(example_task)
print(f"任务 ID: {spec.id}")
print(f"入口点: {spec.entry_point}")
print(f"配置信息:")
for key, value in spec.kwargs.items():
    print(f"  - {key}: {value}")
print()

# 保存完整列表到文件
output_file = "isaac_tasks_list.txt"
with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("IsaacLab 所有已注册的任务列表\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Direct API 任务 ({len(direct_tasks)} 个):\n")
    f.write("-" * 80 + "\n")
    for i, task in enumerate(direct_tasks, 1):
        f.write(f"  {i}. {task}\n")

    f.write(f"\nManager-Based API 任务 ({len(manager_tasks)} 个):\n")
    f.write("-" * 80 + "\n")
    for i, task in enumerate(manager_tasks, 1):
        f.write(f"  {i}. {task}\n")

    f.write(f"\n总计: {len(all_tasks)} 个任务\n")

print(f"完整任务列表已保存到: {output_file}")
print("=" * 80)

# 关闭 simulation app
simulation_app.close()
