import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from backend_engine import SearchEngine


def calculate_ap(retrieved_labels, query_label, top_k=100):
    hits = 0
    sum_precs = 0.0
    for i, lab in enumerate(retrieved_labels[:top_k]):
        if lab == query_label:
            hits += 1
            # 累加当前位置(i+1)的瞬时准确率
            sum_precs += hits / (i + 1.0)

    if hits == 0:
        return 0.0

    return sum_precs / hits


def calculate_precision_at_k(retrieved_ids, relevant_ids, k):
    relevant_set = set(relevant_ids)
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_set)
    return hits / k if k > 0 else 0.0


def evaluate_retrieval(engine, test_samples, l2_to_ids, mode='i2t', top_k=20):
    aps, p_at_10s, p_at_20s = [], [], []
    print(f"\n>>> 开始执行 {mode.upper()} 检索测试 (样本量: {len(test_samples)})...")

    for i, sample in enumerate(test_samples):
        query_id = sample['id']
        query_label = sample['label']
        relevant_ids = l2_to_ids.get(query_label, [])

        try:
            if mode == 'i2t':
                results = engine.search_image(sample['local_path'], top_k=top_k)
            else:
                results = engine.search_text(sample['title'], top_k=top_k)

            retrieved_labels = [res['label'] for res in results]

            hits_at_10 = sum(1 for lab in retrieved_labels[:10] if lab == query_label)
            hits_at_20 = sum(1 for lab in retrieved_labels[:20] if lab == query_label)

            p_at_10s.append(hits_at_10 / 10)
            p_at_20s.append(hits_at_20 / 20)

            current_ap = calculate_ap(retrieved_labels, query_label, top_k=top_k)
            aps.append(current_ap)

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{len(test_samples)}...")

    return {'mAP': np.mean(aps), 'P@10': np.mean(p_at_10s), 'P@20': np.mean(p_at_20s)}


def plot_metrics(metrics_i2t, metrics_t2i, save_path="./results/system_test/acc_test.png"):
    labels = ['mAP@20', 'P@10', 'P@20']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    rects1 = ax.bar(x - width / 2, [metrics_i2t[k] for k in ['mAP', 'P@10', 'P@20']], width, label='I2T (Image->Text)',
                    color='#4C72B0')
    rects2 = ax.bar(x + width / 2, [metrics_t2i[k] for k in ['mAP', 'P@10', 'P@20']], width, label='T2I (Text->Image)',
                    color='#55A868')

    ax.set_ylabel('Score')
    ax.set_title('Retrieval Accuracy Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"\n可视化报告已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    DATA_PATH = "./data/train_info.json"
    print(">>> 正在加载数据集并适配结构...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []
    l2_to_ids = {}

    for product_id, info in raw_data.items():
        item = info.copy()
        item['id'] = product_id
        formatted_data.append(item)

        label = item.get('label', 'default')
        if label not in l2_to_ids:
            l2_to_ids[label] = []
        l2_to_ids[label].append(product_id)

    print(f"成功加载 {len(formatted_data)} 条商品。")

    engine = SearchEngine()
    engine.initialize()

    test_samples = random.sample(formatted_data, min(100, len(formatted_data)))

    res_i2t = evaluate_retrieval(engine, test_samples, l2_to_ids, mode='i2t')
    res_t2i = evaluate_retrieval(engine, test_samples, l2_to_ids, mode='t2i')

    plot_metrics(res_i2t, res_t2i)
