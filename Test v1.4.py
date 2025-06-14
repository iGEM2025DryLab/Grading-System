from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import torch
import logging
import datetime
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. 路径和目录设置 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "Dataset", "data.yaml")

# 创建 model 文件夹
model_parent_dir = os.path.abspath(os.path.join(script_dir, "..", "model"))
os.makedirs(model_parent_dir, exist_ok=True)

# 创建 SavedModels 文件夹，用于存放所有训练记录
output_dir = os.path.join(model_parent_dir, "SavedModels")
os.makedirs(output_dir, exist_ok=True)
logging.info(f"所有训练记录将保存在: {output_dir}")

# 全局最佳模型的路径
overall_best_model_path = os.path.join(output_dir, "best.pt")

# --- 2. 创建本次训练的临时目录 ---
# 稍后会根据训练结果重命名此目录
temp_run_dir = os.path.join(output_dir, f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
os.makedirs(temp_run_dir, exist_ok=True)
logging.info(f"创建本次训练的临时目录: {temp_run_dir}")


# --- 3. 模型训练 ---
# 注意: 'yolo11n.yaml' 可能是一个自定义的名称，标准的YOLOv8 nano模型是 'yolov8n.yaml'
# 如果 'yolo11n.yaml' 是你本地的文件，请确保路径正确。这里使用标准名称作为示例。
model = YOLO("yolov8n.yaml")  # 从YAML构建一个新模型

training_successful = False
try:
    # 训练模型
    model.train(
        data=data_path,
        epochs=15,
        imgsz=640,
        task="detect"
    )
    logging.info("模型训练完成，准备保存和验证结果。")
    training_successful = True
except Exception as e:
    logging.error(f"训练过程中发生错误: {e}")
    training_successful = False

# 等待文件系统同步
import time
time.sleep(2)

# --- 4. 获取、保存和验证本次训练结果 ---
save_dir = None
final_run_dir = None
best_saved_in_run = False
map50_metric = 0.0

if training_successful:
    try:
        # 从 trainer 对象获取ultralytics的保存目录，这是最可靠的方法
        save_dir = str(model.trainer.save_dir)
        logging.info(f"从 Ultralytics 获取到训练输出目录: {save_dir}")
    except Exception as e:
        logging.error(f"无法从 model.trainer 确定训练输出目录: {e}")

if save_dir:
    # 定义源文件路径
    best_src = os.path.join(save_dir, "weights", "best.pt")
    last_src = os.path.join(save_dir, "weights", "last.pt")

    # 定义目标路径（在我们的临时目录中）
    best_dest = os.path.join(temp_run_dir, "best.pt")
    last_dest = os.path.join(temp_run_dir, "last.pt")

    # 复制 best.pt
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dest)
        logging.info(f"已将 best.pt 复制到: {best_dest}")
        best_saved_in_run = True
    else:
        logging.warning(f"源路径中未找到最佳模型: {best_src}")

    # 复制 last.pt
    if os.path.exists(last_src):
        shutil.copy2(last_src, last_dest)
        logging.info(f"已将 last.pt 复制到: {last_dest}")
    else:
        logging.warning(f"源路径中未找到最终模型: {last_src}")

# 如果 best.pt 已成功复制，则进行验证和重命名
if best_saved_in_run:
    try:
        # 加载刚刚复制的模型进行验证
        logging.info(f"加载模型进行验证: {best_dest}")
        trained_model = YOLO(best_dest)
        metrics = trained_model.val(data=data_path) # 确保验证时也指定数据集
        map50_metric = metrics.box.map50
        logging.info(f"模型验证完成。 mAP50: {map50_metric:.4f}")

        # 根据指标重命名目录
        run_number = len(list(Path(output_dir).glob('model_*'))) + 1
        new_dir_name = f"model_{run_number}_map50-{map50_metric:.4f}"
        final_run_dir = os.path.join(output_dir, new_dir_name)
        os.rename(temp_run_dir, final_run_dir)
        logging.info(f"训练目录已重命名为: {final_run_dir}")

    except Exception as e:
        logging.error(f"验证模型或重命名目录时出错: {e}")
        # 如果出错，给一个失败的名称并保留文件以便检查
        final_run_dir = os.path.join(output_dir, f"model_run_failed_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
        os.rename(temp_run_dir, final_run_dir)
else:
    # 如果训练失败或未生成 best.pt，删除空的临时目录
    logging.warning("训练未成功或未生成 best.pt，将删除临时目录。")
    shutil.rmtree(temp_run_dir)


# --- 5. 扫描所有记录并更新全局最佳模型 ---
if final_run_dir:  # 仅当本次运行成功保存后才执行
    try:
        best_run_dir = ""
        highest_metric = -1.0

        # 通过解析目录名称找到最佳模型
        for d in Path(output_dir).glob('model_*'):
            if d.is_dir():
                # 使用正则表达式从文件夹名称中提取map50值
                match = re.search(r'map50-([0-9\.]+)', d.name)
                if match:
                    metric = float(match.group(1))
                    if metric > highest_metric:
                        highest_metric = metric
                        best_run_dir = str(d)

        if best_run_dir:
            logging.info(f"找到历史最佳模型目录: {best_run_dir} (mAP50: {highest_metric:.4f})")
            best_model_source = os.path.join(best_run_dir, "best.pt")
            if os.path.exists(best_model_source):
                shutil.copy2(best_model_source, overall_best_model_path)
                logging.info(f"🎉 已更新全局最佳模型: {overall_best_model_path}")

    except Exception as e:
        logging.error(f"更新全局最佳模型时出错: {e}")


# --- 6. 结果汇总 ---
print("\n" + "="*50)
print(" 训练结果汇总")
print("="*50)

if final_run_dir and os.path.exists(final_run_dir):
    print(f"✅ 本次训练结果已保存至:\n   {final_run_dir}")
    print(f"   - 最佳模型 (mAP50): {map50_metric:.4f}")
else:
    print("❌ 本次训练未能成功保存模型。")

if os.path.exists(overall_best_model_path):
     print(f"\n🏆 当前全局最佳模型已更新/确认:\n   {overall_best_model_path}")
else:
     print("\n- 未能确定或更新全局最佳模型。")
print("="*50)