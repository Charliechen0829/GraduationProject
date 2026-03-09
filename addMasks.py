import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation


def is_image_valid(file_path):
    """
    检查图片文件是否有效：存在、大小不为0、且可以被PIL正常打开
    """
    path = Path(file_path)
    if not path.exists():
        return False
    if os.path.getsize(path) <= 0:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False


def process_and_save_new_path(input_dir, output_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在从本地加载模型: {model_path}")

    model = AutoModelForImageSegmentation.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()

    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    out_path = Path(output_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)

    img_paths = [f for f in Path(input_dir).glob("**/*") if f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
    print(f"📦 找到 {len(img_paths)} 张原始照片，准备开始处理（含损坏检测）...")

    for img_path in tqdm(img_paths, desc="处理进度"):
        try:
            save_name = img_path.stem + ".png"
            target_file_path = out_path / save_name

            # --- 核心改动：检查文件是否存在且完整 ---
            if is_image_valid(target_file_path):
                continue

            # 如果走到这一步，说明文件不存在或者损坏了，执行生成逻辑
            input_img = Image.open(img_path).convert("RGB")
            origin_size = input_img.size

            input_tensor = transform_image(input_img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(input_tensor)[-1].sigmoid().cpu()
                mask_array = preds[0].squeeze().numpy()

            mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8)).resize(origin_size, resample=Image.LANCZOS)
            result_img = input_img.copy()
            result_img.putalpha(mask_pil)

            # 保存并覆盖
            result_img.save(target_file_path, "PNG")

        except Exception as e:
            print(f"❌ 处理 {img_path.name} 出错: {e}")


def update_train_info_json(json_path, new_base_dir):
    """
    修改 JSON 中的路径格式为 E:/thesisData/images/xxx.png
    """
    print(f"📝 正在更新 JSON 路径至: {new_base_dir}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key in data:
        file_id = Path(data[key]['local_path']).stem
        data[key]['local_path'] = f"{new_base_dir}/{file_id}.png".replace("\\", "/")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("✅ JSON 路径更新完成。")


def validate_and_clean_train_info(json_path):
    """
    最终清洗：确保 JSON 里的每一条记录在磁盘上都有对应的、健康的图片
    """
    print("🔍 正在进行最终数据一致性校验...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_data = {}
    for img_id, info in data.items():
        if is_image_valid(info['local_path']):
            valid_data[img_id] = info

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=4, ensure_ascii=False)

    print(f"✨ 清洗完成。原始: {len(data)} 条，剩余有效: {len(valid_data)} 条。")


if __name__ == "__main__":
    # 配置路径
    MODEL_PATH = r".\BiRefNet_Local"
    INPUT_DIR = r".\data\images"
    OUTPUT_DIR = r"E:\thesisData\images"
    JSON_PATH = r"E:\thesisData\train_info.json"

    # 第一步：处理图片（跳过完整文件，重制损坏文件）
    process_and_save_new_path(INPUT_DIR, OUTPUT_DIR, MODEL_PATH)

    # 第二步：更新 JSON 中的路径指向
    update_train_info_json(JSON_PATH, "E:/thesisData/images")

    # 第三步：清洗 JSON，剔除因各种原因最终未生成的坏账
    validate_and_clean_train_info(JSON_PATH)
