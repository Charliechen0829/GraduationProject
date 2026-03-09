import json
import os

from tqdm import tqdm  # 进度条库，如果没有请 pip install tqdm

from backend_engine import engine, CONF


def migrate():
    print(">>> 正在启动迁移程序...")

    # 1. 初始化引擎 (会加载模型、创建/连接数据库、创建FAISS索引)
    # 注意：如果 metadata.db 或 shoh.index 已存在，新数据会被追加到后面
    # 如果想完全重置，请在运行前手动删除 data/metadata.db 和 data/shoh.index
    engine.initialize()

    # 2. 定位原始数据文件
    json_path = os.path.join(CONF['paths']['data_dir'], 'train_info.json')

    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    print(f">>> 读取元数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    print(f">>> 发现 {total_items} 个商品，开始导入...")

    success_count = 0
    fail_count = 0

    # 3. 遍历并导入
    # 使用 tqdm 显示进度条
    for key, item in tqdm(data.items(), desc="Importing"):
        try:
            # --- 路径处理 ---
            # 假设 train_info.json 里的 local_path 是 "images/xxx.jpg" 或者是绝对路径
            # 我们需要拼接成脚本能找到的完整路径
            rel_path = item['local_path']

            # 兼容 Windows/Linux 路径分隔符
            rel_path = rel_path.replace('\\', '/')

            # 拼接完整路径: ./data/images/xxx.jpg
            # 这里的逻辑取决于 train_info.json 里的路径是相对于哪里的
            # 如果 json 里只有文件名，需要 join 'images'
            possible_paths = [
                os.path.join(CONF['paths']['data_dir'], rel_path),  # 方式1: ./data/images/001.jpg
                os.path.join(CONF['paths']['data_dir'], 'images', rel_path),  # 方式2: ./data/images/images/001.jpg
                rel_path  # 方式3: 绝对路径
            ]

            full_image_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    full_image_path = p
                    break

            if not full_image_path:
                # 如果还是找不到，打印一条日志看看到底搜的是什么路径，方便调试
                # print(f"找不到图片: {rel_path}, 尝试过: {possible_paths[0]}")
                fail_count += 1
                continue

            # 检查文件是否存在
            if not os.path.exists(full_image_path):
                # 尝试另一种路径组合（容错处理）
                full_image_path = os.path.join(CONF['paths']['data_dir'], rel_path)
                if not os.path.exists(full_image_path):
                    # print(f"警告: 图片文件丢失 {full_image_path}")
                    fail_count += 1
                    continue

            # --- 品牌提取逻辑 (与之前 app.py 保持一致) ---
            labels_raw = item.get('label', '')
            brand = 'Other'
            if labels_raw:
                # 取分号前的第一段，再取空格前的第一词
                parts = labels_raw.split(';')[0].strip().split(' ', 1)
                if parts and parts[0]:
                    brand = parts[0]

            title = item.get('title', 'Unknown Product')

            # --- 调用引擎核心方法 ---
            # 这一步会：1. 提取特征 2. 计算哈希 3. 存入 SQLite 4. 存入 FAISS
            engine.add_product(
                image_path=full_image_path,
                title=title,
                brand=brand,
                label=labels_raw
            )

            success_count += 1

        except Exception as e:
            print(f"Error processing {key}: {e}")
            fail_count += 1

    print("\n" + "=" * 30)
    print(f"迁移完成!")
    print(f"成功导入: {success_count}")
    print(f"失败/跳过: {fail_count}")
    print(f"数据库位置: {CONF['paths']['database_file']}")
    print(f"索引位置: {CONF['paths']['faiss_index_file']}")
    print("=" * 30)


if __name__ == "__main__":
    migrate()
