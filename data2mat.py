import json
import os

import numpy as np
import scipy.io as sio

from utils import EnhancedFeatureExtractor


def compute_a1_2(L1, L2):
    """
    计算层间关联矩阵 A1_2
    公式: A1_2 = pinv(L1.T * L1) * L1.T * L2
    """
    print("正在计算层间关联矩阵 A1_2...")
    try:
        # 使用伪逆矩阵计算，防止奇异矩阵错误
        A1_2 = np.linalg.pinv(L1.T @ L1) @ L1.T @ L2
        return A1_2
    except Exception as e:
        print(f"计算 A1_2 失败: {e}")
        return np.zeros((L1.shape[1], L2.shape[1]))


def main():
    # ================= 配置路径 =================
    # JSON 文件路径
    json_path = r'E:/thesisData/train_info.json'
    # 输出的 .mat 文件路径
    output_mat_path = r'E:/thesisData/custom_data.mat'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_mat_path), exist_ok=True)

    # ================= 1. 加载数据 =================
    print(f"正在加载 JSON 数据: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = []
    l1_set = set()
    l2_set = set()

    # 解析 JSON
    print("正在解析标签和路径...")
    for key, value in data.items():
        # 处理路径分隔符
        full_image_path = os.path.join(value['local_path']).replace("\\", "/")

        item = {
            'id': key,
            'image_path': full_image_path,
            'title': value['title'],
            'L1': [],
            'L2': []
        }

        # 解析标签逻辑 (参考 utils.py)
        labels_raw = value.get('label', '')
        for label_segment in labels_raw.split(';'):
            if not label_segment.strip():
                continue
            parts = label_segment.strip().split(' ', 1)
            l1_brand = parts[0]
            # 如果没有第二部分，则由品牌名充当产品名
            l2_product = parts[1] if len(parts) > 1 else l1_brand

            item['L1'].append(l1_brand)
            l1_set.add(l1_brand)
            item['L2'].append(l2_product)
            l2_set.add(l2_product)

        # 处理无标签情况
        if not item['L1']:
            item['L1'].append('other')
            l1_set.add('other')
        if not item['L2']:
            item['L2'].append('other')
            l2_set.add('other')

        items.append(item)

    print(f"共加载 {len(items)} 条数据")

    # ================= 2. 提取特征 =================
    # 初始化特征提取器 (来自 utils.py)
    extractor = EnhancedFeatureExtractor()

    image_paths = [item['image_path'] for item in items]
    texts = [item['title'] for item in items]

    # 提取图像特征 (Image/Gist)
    print("开始提取图像特征...")
    X = extractor.extract_image_features_capture(image_paths)

    # 提取文本特征 (Tag/Text)
    print("开始提取文本特征...")
    Y = extractor.extract_text_features_capture(texts)

    # ================= 3. 构建标签矩阵 =================
    print("构建标签矩阵...")
    l1_list = sorted(list(l1_set))
    l2_list = sorted(list(l2_set))

    l1_to_idx = {lbl: i for i, lbl in enumerate(l1_list)}
    l2_to_idx = {lbl: i for i, lbl in enumerate(l2_list)}

    num_samples = len(items)
    num_class1 = len(l1_list)
    num_class2 = len(l2_list)

    L1_mat = np.zeros((num_samples, num_class1))
    L2_mat = np.zeros((num_samples, num_class2))

    for i, item in enumerate(items):
        for lbl in item['L1']:
            L1_mat[i, l1_to_idx[lbl]] = 1
        for lbl in item['L2']:
            L2_mat[i, l2_to_idx[lbl]] = 1

    # 合并标签 (MATLAB load_dataset.m 通常需要这种形式)
    # 某些数据集如 FashionVC 是直接使用 Label 矩阵，然后代码里拆分
    # 这里我们保存 L1 和 L2，同时也保存合并的 Label 以兼容不同读取方式
    Label_concat = np.hstack((L1_mat, L2_mat))

    # ================= 4. 计算关联矩阵 A1_2 =================
    A1_2 = compute_a1_2(L1_mat, L2_mat)

    # ================= 5. 保存 .mat 文件 =================
    mat_data = {
        'Image': X,  # 图像特征
        'Tag': Y,  # 文本特征
        'L1': L1_mat,  # 第一层标签 (可选)
        'L2': L2_mat,  # 第二层标签 (可选)
        'Label': Label_concat,  # 合并标签 [L1, L2]
        'A1_2': A1_2,  # 层间关联矩阵
        # 额外保存一些元数据方便调试
        'num_class1': num_class1,
        'num_class2': num_class2,
        'L1_names': l1_list,
        'L2_names': l2_list
    }

    print(f"正在保存数据到: {output_mat_path}")
    print(f"数据维度信息:")
    print(f"  Image: {X.shape}")
    print(f"  Tag:   {Y.shape}")
    print(f"  Label: {Label_concat.shape} (L1: {num_class1} + L2: {num_class2})")
    print(f"  A1_2:  {A1_2.shape}")

    sio.savemat(output_mat_path, mat_data)
    print("完成！")


if __name__ == '__main__':
    main()
