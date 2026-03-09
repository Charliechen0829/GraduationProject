import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_complex_distribution(json_path, save_dir='./results/'):
    # 1. 加载数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有类别 (Category)
    categories = []
    for item in data.values():
        label_str = item.get('label', '')
        # 按照你的习惯提取具体品类
        cat = label_str.split('/')[-1] if '/' in label_str else label_str.split(' ')[-1]
        categories.append(cat)

    # 2. 统计数据
    counts_map = Counter(categories)

    # 获取原始顺序的列表 (按 ID 顺序，这里假设 ID 是提取出的 cat 字符串)
    # 如果你的 ID 是数字，可以增加一次 sorted(counts_map.keys())
    raw_ids = list(counts_map.keys())
    raw_values = [counts_map[k] for k in raw_ids]

    # 获取降序排列的数据 (用于绘制长尾折线)
    sorted_values = sorted(raw_values, reverse=True)

    # 3. 绘图
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)

    # --- 绘制：不排序的极细柱状图 ---
    x_raw = np.arange(len(raw_values))
    # 使用随机/循环色系，让不排序的柱子看起来更有“多样性”
    colors = plt.cm.tab20(np.linspace(0, 1, len(raw_values) % 20 + 20))
    ax1.bar(x_raw, raw_values, color=colors, width=1.0, alpha=0.7)

    ax1.set_xlabel('Category ID', fontsize=12)
    ax1.set_ylabel('Sample Number', fontsize=12)
    ax1.set_title('Data Distribution', fontsize=14, fontweight='bold')

    # --- 绘制：降序排列的长尾折线 ---
    ax2 = ax1.twinx()
    x_sorted = np.arange(len(sorted_values))
    ax2.plot(x_sorted, sorted_values, color='#E31A1C', linewidth=2)

    # 填充折线下方，强化长尾视觉效果
    # ax2.fill_between(x_sorted, sorted_values, color='#E31A1C', alpha=0.1)

    ax2.set_ylabel('Frequency', fontsize=12, color='#E31A1C')
    ax2.tick_params(axis='y', labelcolor='#E31A1C')

    # 4. 细节调整
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'data_distribution.png')
    plt.savefig(save_path)
    print(f"图表已生成并保存至: {save_path}")
    # plt.show()


def plot_l1_distribution(json_path, save_dir='./results/'):
    # 1. 数据加载
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    l1_categories = []
    for item in data.values():
        label_str = item.get('label', '')
        # 逻辑：提取 L1。假设格式为 "L1/L2" 或 "L1 L2"
        # 如果没有斜杠，则认为整个标签就是 L1
        l1_cat = label_str.split('/')[0] if '/' in label_str else label_str.split(' ')[0]
        l1_categories.append(l1_cat)

    # 2. 统计
    counts_map = Counter(l1_categories)

    # 获取自然顺序（可以按字母排序或 JSON 出现的先后顺序）
    raw_l1_names = list(counts_map.keys())
    raw_values = [counts_map[k] for k in raw_l1_names]

    # 获取降序排列用于长尾折线
    sorted_values = sorted(raw_values, reverse=True)

    # 3. 绘图
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # --- 柱状图：按 L1 原始/字母顺序 ---
    x_raw = np.arange(len(raw_values))
    # 为 L1 类别提供更鲜艳的配色
    colors = plt.cm.Set3(np.linspace(0, 1, len(raw_values)))

    # 这里的 width 适当调大，因为 L1 类目通常没那么多
    ax1.bar(x_raw, raw_values, color=colors, width=0.7, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax1.set_xlabel('L1 Categories', fontsize=12)
    ax1.set_ylabel('Sample Number', fontsize=12)
    ax1.set_title('L1 Level Label Distribution', fontsize=14, fontweight='bold')

    # --- 折线图：降序长尾 ---
    ax2 = ax1.twinx()
    x_sorted = np.arange(len(sorted_values))
    ax2.plot(x_sorted, sorted_values, color='#D62728', marker='o', markersize=4, linewidth=2,
             )

    ax2.set_ylabel('Frequency', fontsize=12, color='#D62728')
    ax2.tick_params(axis='y', labelcolor='#D62728')

    # 4. 细节微调
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    # 如果 L1 类别不多，可以考虑把名字印在横轴上（可选）
    if len(raw_l1_names) < 20:
        ax1.set_xticks(x_raw)
        ax1.set_xticklabels(raw_l1_names, rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'L1_distribution.png')
    plt.savefig(save_path)
    print(f"L1 分布图已保存至: {save_path}")
    # plt.show()


def analyze_train_dataset(json_path):
    """
    根据标签分隔符统计多品样本及类别总数
    """
    if not os.path.exists(json_path):
        print(f"错误: 未找到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_samples = len(data)
    single_product_count = 0
    multi_product_count = 0
    all_categories = set()
    unique_brands = set()

    for item_id, info in data.items():
        # 获取标签字符串，例如 "AHC 黄金面霜;AHC 黄金水"
        label_str = info.get('label', '')

        # 1. 处理类别统计：按分号分割并去重
        # 即使是单品，分割后也是包含一个元素的列表
        item_labels = [l.strip() for l in label_str.split(';') if l.strip()]
        for label in item_labels:
            all_categories.add(label)
            # 假设品牌是标签中的第一个词（如 "AHC"），进行简单提取统计
            brand_guess = label.split(' ')[0]
            unique_brands.add(brand_guess)

        # 2. 统计多品 vs 单品
        # 根据要求，包含分号且分割后元素大于1的为多品样本
        if len(item_labels) > 1:
            multi_product_count += 1
        else:
            single_product_count += 1

    # 打印统计结果
    print("\n" + "=" * 40)
    print("      Product1M 训练集子集详细统计报告")
    print("=" * 40)
    print(f"总样本数 (Total Samples):          {total_samples}")
    print(f"单品样本数 (Single-product):        {single_product_count}")
    print(f"多品样本数 (Multi-product):         {multi_product_count}")
    print(f"总类别数 (Total Categories):       {len(all_categories)}")
    print(f"推算品牌数 (Estimated Brands):      {len(unique_brands)}")

    multi_ratio = (multi_product_count / total_samples) * 100 if total_samples > 0 else 0
    print(f"多品样本占比 (Multi-product %):     {multi_ratio:.2f}%")
    print("=" * 40)

    # 可选：打印前5个类别示例
    if all_categories:
        print(f"类别示例: {list(all_categories)[:5]}...")
    print("=" * 40 + "\n")


def visualize_processing_comparison(original_dir, processed_dir, save_dir='./results', n=3):
    original_dir = os.path.abspath(original_dir)
    processed_dir = os.path.abspath(processed_dir)

    if original_dir.lower() == processed_dir.lower():
        print("警告：原始路径和处理后路径相同！")
        return

    # 获取原始图像文件列表
    valid_exts = ('.jpg', '.jpeg', '.png')
    original_files = [f for f in os.listdir(original_dir) if f.lower().endswith(valid_exts)]

    processed_files = os.listdir(processed_dir)
    processed_map = {os.path.splitext(f)[0]: f for f in processed_files if f.lower().endswith(valid_exts)}
    common_names = [os.path.splitext(f)[0] for f in original_files if os.path.splitext(f)[0] in processed_map]

    if not common_names:
        print("未发现同名匹配文件，请检查目录。")
        return

    selected_names = random.sample(common_names, min(n, len(common_names)))
    num_samples = len(selected_names)

    # --- 修改部分：从 n 行 2 列 (n, 2) 转为 2 行 n 列 (2, n) ---
    # 第一行全为 Original，第二行全为 Processed
    fig, axes = plt.subplots(2, num_samples, figsize=(5 * num_samples, 10))

    for i, name in enumerate(selected_names):
        # 获取完整路径
        orig_file = [f for f in original_files if os.path.splitext(f)[0] == name][0]
        proc_file = processed_map[name]
        p1 = os.path.join(original_dir, orig_file)
        p2 = os.path.join(processed_dir, proc_file)
        img1 = Image.open(p1)
        img2 = Image.open(p2)

        # 处理只有 1 个样本时 axes 维度退化的问题
        if num_samples == 1:
            ax_top = axes[0]
            ax_bottom = axes[1]
        else:
            ax_top = axes[0, i]  # 第一行第 i 列
            ax_bottom = axes[1, i]  # 第二行第 i 列

        # 展示原图 (第一行)
        ax_top.imshow(img1)
        ax_top.set_title(f"Original\n({orig_file})")
        ax_top.axis('off')

        # 展示处理后的图 (第二行)
        ax_bottom.imshow(img2)
        ax_bottom.set_title(f"Processed\n({proc_file})")
        ax_bottom.axis('off')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'compare_mask.png')
    plt.savefig(save_path)


if __name__ == '__main__':
    train_json_path = r'./data/train_info.json'
    raw_img_path = r'D:\github\thesis\data\images'
    processed_img_path = r'E:\thesisData\images'
    #
    # plot_complex_distribution(train_json_path)
    # plot_l1_distribution(train_json_path)
    # analyze_train_dataset(train_json_path)

    visualize_processing_comparison(raw_img_path, processed_img_path, n=3)
