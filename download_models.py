# download_assets.py
import os

from transformers import ViTModel, BertModel, AutoTokenizer, CLIPModel, CLIPProcessor

# 定义保存路径
MODEL_ROOT = './resources/models'
os.makedirs(MODEL_ROOT, exist_ok=True)


def download_and_save(model_class, tokenizer_class, model_name, save_name):
    save_path = os.path.join(MODEL_ROOT, save_name)
    print(f"正在下载并保存 {model_name} 到 {save_path} ...")

    # 下载模型
    model = model_class.from_pretrained(model_name)
    model.save_pretrained(save_path)

    if tokenizer_class:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)

    print(f"Successfully saved {save_name}!")


if __name__ == '__main__':
    print(">>> 开始下载离线模型资源...")

    download_and_save(ViTModel, None, "google/vit-base-patch16-224", "vit-base")

    download_and_save(BertModel, AutoTokenizer, "bert-base-uncased", "bert-base")

    download_and_save(CLIPModel, CLIPProcessor, "openai/clip-vit-base-patch32", "clip-base")

    print("\n>>> 所有模型已下载完毕！现在可以断网运行了。")
