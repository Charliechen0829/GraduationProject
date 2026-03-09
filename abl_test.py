import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from backend_engine import SearchEngine

# 初始化引擎
engine = SearchEngine()
engine.initialize()


def calculate_ap_by_label(retrieved_labels, query_label, top_k=20):
    """
    修正后的 AP 计算逻辑：
    AP = (1 / 命中总数) * sum(第 i 个命中位置的 Precision)
    """
    hits = 0
    sum_precs = 0.0

    # 遍历前 top_k 个检索结果
    for i, lab in enumerate(retrieved_labels[:top_k]):
        if lab == query_label:
            hits += 1
            # 计算当前位置的 Precision = (当前命中数 / 当前排名)
            sum_precs += hits / (i + 1.0)

    # 标准定义：分母应该是检索列表中相关条目的总数 (hits)
    # 如果在该 top_k 范围内一个都没中，则 AP 为 0
    return sum_precs / hits if hits > 0 else 0.0


def calculate_recall_by_label(retrieved_labels, query_label, k):
    # 计算 Recall@K：前 K 个结果中是否有至少一个 label 匹配
    for lab in retrieved_labels[:k]:
        if lab == query_label:
            return 1.0
    return 0.0


def strategy_plan_a(query_text, top_k=20):
    start_time = time.time()
    results = engine.search_text(query_text, top_k=top_k)
    duration = time.time() - start_time
    return results, duration


def strategy_plan_b(query_text, top_k_display=20, candidate_pool_size=500):
    start_time = time.time()
    # 向量召回
    candidates = engine.search_text(query_text, top_k=candidate_pool_size)

    exact_matches = []
    semantic_matches = []
    query_lower = query_text.lower()

    # 内存重排序与文本过滤
    for item in candidates:
        item_title = item.get('title', '').lower()
        if query_lower in item_title:
            exact_matches.append(item)
        else:
            semantic_matches.append(item)

    # 组合结果
    results = (exact_matches + semantic_matches)[:top_k_display]
    duration = time.time() - start_time
    return results, duration


def run_ablation_test(test_samples, top_k=20):
    metrics = {
        'plan_a': {'aps': [], 'times': [], 'recalls': {k: [] for k in [1, 5, 10, 20]}},
        'plan_b': {'aps': [], 'times': [], 'recalls': {k: [] for k in [1, 5, 10, 20]}}
    }

    print(f"\n>>> 开始 T2I 消融实验对比 (样本量: {len(test_samples)})...")

    for i, sample in enumerate(test_samples):
        query_text = sample['title']
        query_label = sample['label']

        # A
        res_a, dur_a = strategy_plan_a(query_text, top_k=top_k)
        labels_a = [r['label'] for r in res_a]
        metrics['plan_a']['aps'].append(calculate_ap_by_label(labels_a, query_label, top_k))
        metrics['plan_a']['times'].append(dur_a)
        for k in [1, 5, 10, 20]:
            metrics['plan_a']['recalls'][k].append(calculate_recall_by_label(labels_a, query_label, k))

        # B
        res_b, dur_b = strategy_plan_b(query_text, top_k_display=top_k, candidate_pool_size=1000)
        labels_b = [r['label'] for r in res_b]
        metrics['plan_b']['aps'].append(calculate_ap_by_label(labels_b, query_label, top_k))
        metrics['plan_b']['times'].append(dur_b)
        for k in [1, 5, 10, 20]:
            metrics['plan_b']['recalls'][k].append(calculate_recall_by_label(labels_b, query_label, k))

        if (i + 1) % 10 == 0:
            print(f"进度: {i + 1}/{len(test_samples)}...")

    return metrics


def plot_ablation_report(metrics):
    # 均值计算
    map_a, map_b = np.mean(metrics['plan_a']['aps']), np.mean(metrics['plan_b']['aps'])
    time_a, time_b = np.mean(metrics['plan_a']['times']) * 1000, np.mean(metrics['plan_b']['times']) * 1000

    plt.figure(figsize=(15, 6))

    # Recall@K 曲线
    plt.subplot(1, 2, 1)
    ks = [1, 5, 10, 20]
    rec_a = [np.mean(metrics['plan_a']['recalls'][k]) for k in ks]
    rec_b = [np.mean(metrics['plan_b']['recalls'][k]) for k in ks]
    plt.plot(ks, rec_a, 'o--', label='Plan A (Base)', color='#ff7f0e', linewidth=2)
    plt.plot(ks, rec_b, 's-', label='Plan B (Enhanced)', color='#1f77b4', linewidth=2)
    plt.title('Recall@K Comparison (Ablation Study)', fontsize=12)
    plt.xlabel('K')
    plt.ylabel('Recall Rate (Label Match)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Latency vs. Accuracy 散点图
    plt.subplot(1, 2, 2)
    plt.scatter(time_a, map_a, color='#ff7f0e', s=200, label='Plan A')
    plt.scatter(time_b, map_b, color='#1f77b4', s=200, label='Plan B')
    plt.annotate('', xy=(time_b, map_b), xytext=(time_a, map_a),
                 arrowprops=dict(arrowstyle="->", color='gray', lw=1.5, alpha=0.5))

    plt.text(time_a, map_a + 0.005, f'Base ({time_a:.1f}ms)', ha='center', fontsize=10)
    plt.text(time_b, map_b + 0.005, f'Enh. ({time_b:.1f}ms)', ha='center', fontsize=10)

    plt.title('Latency vs. Accuracy Trade-off', fontsize=12)
    plt.xlabel('Average Latency (ms)')
    plt.ylabel('mAP (Label Match)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("./results/system_test/ablation_test_report.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # 加载数据
    DATA_PATH = "./data/train_info.json"
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []
    for product_id, info in raw_data.items():
        item = info.copy()
        item['id'] = product_id
        formatted_data.append(item)

    # 抽取 1% 样本测试
    test_samples = random.sample(formatted_data, min(370, len(formatted_data)))

    # 运行消融实验
    results_metrics = run_ablation_test(test_samples)

    # 打印最终对比结果
    print("\n" + "=" * 50)
    print(f"{'指标 (Avg)':<15} | {'Plan A (Base)':<15} | {'Plan B (Enh)':<15}")
    print("-" * 50)
    print(
        f"{'mAP':<15} | {np.mean(results_metrics['plan_a']['aps']):.4f} | {np.mean(results_metrics['plan_b']['aps']):.4f}")
    print(
        f"{'Recall@20':<15} | {np.mean(results_metrics['plan_a']['recalls'][20]):.4f} | {np.mean(results_metrics['plan_b']['recalls'][20]):.4f}")
    print(
        f"{'Latency':<15} | {np.mean(results_metrics['plan_a']['times']) * 1000:.2f} ms | {np.mean(results_metrics['plan_b']['times']) * 1000:.2f} ms")
    print("=" * 50)

    # 绘图
    plot_ablation_report(results_metrics)
