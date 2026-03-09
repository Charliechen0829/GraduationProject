import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from backend_engine import SearchEngine

# 1. 初始化引擎
engine = SearchEngine()
engine.initialize()


def calculate_strict_ap(retrieved_labels, query_label, k=20):
    hits = 0
    sum_precs = 0.0
    for i, lab in enumerate(retrieved_labels[:k]):
        if lab == query_label:
            hits += 1
            # 瞬时精度 = 当前命中总数 / 当前名次
            sum_precs += hits / (i + 1.0)

    if hits == 0:
        return 0.0
    return sum_precs / hits


def calculate_topk_overlap(paths_before, paths_after):
    """
    Top-K 集合重合度 (Jaccard Similarity)。
    用于衡量新商品注入后，原有查询的邻域结构被改变了多少。
    """
    set_before = set(paths_before)
    set_after = set(paths_after)
    intersection = len(set_before.intersection(set_after))
    union = len(set_before.union(set_after))
    return intersection / union if union > 0 else 0.0


def run_dynamic_update_test(num_new_items=100, k_val=20):
    # 加载原始数据
    data_path = "./data/train_info.json"
    if not os.path.exists(data_path):
        data_path = r"E:/thesisData/train_info.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        all_data = list(raw_data.values())

    # 随机挑选新上架商品与用于观察的老商品
    new_items = random.sample(all_data, num_new_items)
    old_items = random.sample(all_data, 50)

    results = {
        'update_times': [],
        'nihr_hits': 0,
        'map_before': 0,
        'map_after': 0,
        'overlap_scores': []
    }

    print(f">>> 阶段 1: 记录更新前旧数据的 mAP 与排序快照 (K={k_val})...")
    pre_map_list = []
    snapshot_before = {}
    for item in old_items:
        res = engine.search_image(item['local_path'], top_k=k_val)
        labels = [r['label'] for r in res]
        paths = [r['image_path'].replace('\\', '/') for r in res]

        pre_map_list.append(calculate_strict_ap(labels, item['label'], k=k_val))
        snapshot_before[item['local_path']] = paths

    results['map_before'] = np.mean(pre_map_list)

    print(f">>> 阶段 2: 正在上架 {num_new_items} 个新品...")
    for item in new_items:
        start_t = time.time()
        # 写入数据库并更新 FAISS 索引
        engine.add_product(
            image_path=item['local_path'],
            title=item['title'],
            brand=item.get('brand', 'Unknown'),
            label_raw=item['label']
        )
        results['update_times'].append((time.time() - start_t) * 1000)

        # 验证新商品立即召回能力 (NIHR)
        res_nihr = engine.search_image(item['local_path'], top_k=1)
        if res_nihr and res_nihr[0]['image_path'].replace('\\', '/') == item['local_path'].replace('\\', '/'):
            results['nihr_hits'] += 1

    print(f">>> 阶段 3: 对比更新后旧数据的 mAP 变化及索引扰动...")
    post_map_list = []
    for item in old_items:
        res = engine.search_image(item['local_path'], top_k=k_val)
        labels = [r['label'] for r in res]
        paths_after = [r['image_path'].replace('\\', '/') for r in res]

        # 计算新 mAP
        post_map_list.append(calculate_strict_ap(labels, item['label'], k=k_val))

        # 计算排序重合度
        paths_before = snapshot_before[item['local_path']]
        overlap = calculate_topk_overlap(paths_before, paths_after)
        results['overlap_scores'].append(overlap)

    results['map_after'] = np.mean(post_map_list)
    results['nihr'] = results['nihr_hits'] / num_new_items
    results['avg_overlap'] = np.mean(results['overlap_scores'])

    return results


def plot_comprehensive_results(res):
    fig = plt.figure(figsize=(15, 5))

    # mAP 精度微变对比
    ax1 = plt.subplot(1, 3, 1)
    labels = ['Before Update', 'After Update']
    values = [res['map_before'], res['map_after']]
    bars = ax1.bar(labels, values, color=['#34495e', '#3498db'], width=0.4)
    ax1.set_ylim(0, 1.2)
    ax1.set_title("mAP Stability", fontsize=12)
    ax1.set_ylabel("Strict mAP@20")
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f'{h:.5f}', ha='center', fontweight='bold')

    # 索引结构重合度
    ax2 = plt.subplot(1, 3, 2)
    overlap_pct = res['avg_overlap'] * 100
    ax2.bar(['Ranking Overlap'], [overlap_pct], color='#9b59b6', width=0.3)
    ax2.set_ylim(0, 110)
    ax2.set_title("Index Structure Preservation", fontsize=12)
    ax2.set_ylabel("Top-K Jaccard Overlap (%)")
    ax2.text(0, overlap_pct + 2, f"{overlap_pct:.2f}%", ha='center', fontweight='bold')

    # 在图内附加 NIHR 文本信息
    ax2.text(0, overlap_pct / 2, f"NIHR: {res['nihr'] * 100:.1f}%\n(New Item Hit Rate)",
             ha='center', va='center', color='white', fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    # 更新耗时监控
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(res['update_times'], color='#e67e22', marker='.', alpha=0.8)
    avg_t = np.mean(res['update_times'])
    ax3.axhline(y=avg_t, color='r', linestyle='--', label=f'Avg: {avg_t:.1f}ms')
    ax3.set_title("Real-time Indexing Latency", fontsize=12)
    ax3.set_xlabel("Item Sequence")
    ax3.set_ylabel("Latency (ms)")
    ax3.legend()

    plt.tight_layout()
    os.makedirs("./results", exist_ok=True)
    plt.savefig("./results/system_test/update_test.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    test_res = run_dynamic_update_test(num_new_items=100, k_val=20)

    print("\n" + "=" * 60)
    print(f"【索引动态维护与特征包容性验证报告】")
    print(f"1. NIHR (新商品首位命中率) : {test_res['nihr'] * 100:.2f}%  -> [验证即时生效能力]")
    print(f"2. mAP_before (更新前精度) : {test_res['map_before']:.5f}")
    print(f"3. mAP_after  (更新后精度) : {test_res['map_after']:.5f}")
    print(f"4. mAP 波动幅度 (Delta P)  : {(test_res['map_after'] - test_res['map_before']):.5f}")
    print(f"5. Top-K 排序重合度        : {test_res['avg_overlap'] * 100:.2f}% -> [衡量原邻域结构的被扰动率]")
    print(f"6. 平均单件商品更新耗时    : {np.mean(test_res['update_times']):.2f} ms")
    print("=" * 60)

    plot_comprehensive_results(test_res)
