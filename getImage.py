import json
import os
import random
import time

import requests


def download_image(url, save_path):
    # 模拟浏览器请求头，避免被服务器拦截
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        # 添加超时和请求头参数
        response = requests.get(url, stream=True, timeout=10, headers=headers)
        response.raise_for_status()  # 检查请求是否成功

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def process_file(filename, image_dir, train_info):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    with open(filename, 'r', encoding='utf-8') as f:
        origin_line = f.readlines()
        lines = origin_line[:]
        total_count = len(lines)

        for idx, line in enumerate(lines):
            parts = line.strip().split('#####')
            if len(parts) < 5:
                continue

            product_id = parts[0]
            title = parts[1]
            image_url = parts[2]
            label = parts[-1]  # 最后一个字段作为label

            # 处理label中的分隔符，将"#;#"替换为";"
            label = label.replace("#;#", ";")

            # 处理文件扩展名
            file_ext = image_url.split('.')[-1].lower()
            if file_ext not in ['jpg', 'jpeg', 'png', 'gif']:
                file_ext = 'jpg'

            # 使用相对路径
            relative_path = os.path.join("images", f"{product_id}.{file_ext}")
            save_path = os.path.join(image_dir, f"{product_id}.{file_ext}")

            print(f"Downloading {idx + 1}/{total_count}: {image_url}")

            # 检查图片是否已存在
            image_exists = os.path.exists(save_path)

            if not image_exists:
                # 添加随机延迟，避免请求过于频繁
                time.sleep(random.uniform(0.5, 2.0))

                success = download_image(image_url, save_path)

                # 如果下载失败，额外延迟
                if not success:
                    time.sleep(3)
                    continue  # 跳过添加到train_info
            else:
                print(f"Image already exists: {save_path}")

            # 添加到train_info，无论图片是新下载的还是已存在的
            modified_relative_path = relative_path.replace('\\', '/')

            # 添加到train_info，无论图片是新下载的还是已存在的
            train_info[product_id] = {
                "url": image_url,
                "title": title,
                "local_path": f"./data/{modified_relative_path}",
                "label": label
            }

            # 每处理50个图片就保存一次JSON，防止程序中断导致数据丢失
            if (idx + 1) % 50 == 0:
                with open('data/train_info.json', 'w', encoding='utf-8') as f:
                    json.dump(train_info, f, ensure_ascii=False, indent=2)
                print(f"Checkpoint: Saved train_info.json after {idx + 1} images")


if __name__ == '__main__':
    # files = ['./data/product1m_dev_ossurl_v2.txt', './data/product1m_gallery_ossurl_v2.txt'] # 已处理完套装商品
    files = ['./data/product1m_gallery_ossurl_v2.txt']
    image_dir = './data/images'

    # 确保数据目录存在
    os.makedirs('./data', exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # 加载或初始化train_info
    train_info = {}
    json_path = 'data/train_info.json'
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                train_info = json.load(f)
            print("Loaded existing train_info.json")
        except Exception as e:
            print(f"Error loading train_info.json: {e}")
            train_info = {}

    for file in files:
        if os.path.exists(file):
            process_file(file, image_dir, train_info)
        else:
            print(f"File {file} not found.")

    # 保存train_info.json
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)
    print(f"Final: Saved train_info.json with {len(train_info)} entries")
