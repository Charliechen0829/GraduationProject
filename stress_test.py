import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

from backend_engine import SearchEngine

engine = SearchEngine()
engine.initialize()


def load_test_data(data_path="./data/train_info.json"):
    if not os.path.exists(data_path):
        return []
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_list = []
    for product_id, info in raw_data.items():
        img_rel_path = info.get('local_path') or info.get('image_path')
        if not img_rel_path: continue

        full_path = os.path.join("./data/", img_rel_path)
        if not os.path.exists(full_path):
            full_path = img_rel_path  # 尝试作为相对根目录的路径

        if os.path.exists(full_path):
            data_list.append(full_path)

    print(f">>> 数据加载完成，共找到 {len(data_list)} 张有效图片。")
    return data_list


def task_worker(img_path):
    # 模拟单次搜索任务
    start_time = time.time()
    try:
        _ = engine.search_image(img_path, top_k=20)
        return (time.time() - start_time) * 1000  # ms
    except:
        return None


def run_stress_test(concurrency, test_samples, num_requests=100):
    print(f"\n>>> 启动并发测试: 并发数={concurrency}, 总请求={num_requests}...")
    current_samples = [random.choice(test_samples) for _ in range(num_requests)]

    start_wall = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        latencies = list(executor.map(task_worker, current_samples))
    end_wall = time.time()

    valid_latencies = [l for l in latencies if l is not None]
    if not valid_latencies:
        return None

    duration = end_wall - start_wall
    return {
        "concurrency": concurrency,
        "qps": len(valid_latencies) / duration,
        "avg": np.mean(valid_latencies),
        "p50": np.percentile(valid_latencies, 50),
        "p95": np.percentile(valid_latencies, 95),
        "p99": np.percentile(valid_latencies, 99),
        "raw": valid_latencies
    }


def visualize_results(results):
    """
    绘制测试报告
    1. 响应耗时分布（饼图 - 自动适配最大延迟）
    2. 并发 vs QPS (折线图)
    3. 并发 vs 响应时间 (柱状图)
    """
    # 取最高并发级别的数据
    max_res = results[-1]
    lats = np.array(max_res['raw'])

    p50 = max_res['p50']
    p99 = max_res['p99']

    # 自动定义四个区间：极速、正常、缓慢、极慢
    # 使用 P50 和 P99 作为分界参考点，确保饼图总是有比例显示
    b1 = p50 * 0.5  # 优秀水平线
    b2 = p50  # 中位数线
    b3 = (p50 + p99) / 2  # 过渡线

    bins = [0, b1, b2, b3, max(b3 + 1, p99 * 1.2)]
    labels = [
        f'Fast (<{b1:.0f}ms)',
        f'Normal ({b1:.0f}-{b2:.0f}ms)',
        f'Slow ({b2:.0f}-{b3:.0f}ms)',
        f'Very Slow (>{b3:.0f}ms)'
    ]

    counts = [len(lats[(lats >= bins[i]) & (lats < bins[i + 1])]) for i in range(len(labels))]
    final_counts = []
    final_labels = []
    for c, l in zip(counts, labels):
        if c > 0:
            final_counts.append(c)
            final_labels.append(l)

    plt.figure(figsize=(16, 6))

    # 子图1：饼图 (耗时分布)
    plt.subplot(1, 3, 1)
    if final_counts:
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        plt.pie(final_counts, labels=final_labels, autopct='%1.1f%%',
                startangle=140, colors=colors[:len(final_counts)],
                pctdistance=0.85, explode=[0.05] * len(final_counts))
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        plt.gca().add_artist(centre_circle)
    plt.title(f"Latency Distribution\n(Conc: {max_res['concurrency']})", fontsize=12)

    # 子图2：QPS 趋势
    plt.subplot(1, 3, 2)
    concs = [r['concurrency'] for r in results]
    qps_vals = [r['qps'] for r in results]
    plt.plot(concs, qps_vals, marker='o', lw=2, color='#2980b9')
    plt.fill_between(concs, qps_vals, alpha=0.1, color='#2980b9')
    plt.xlabel('Concurrency')
    plt.ylabel('QPS (Queries Per Second)')
    plt.title('System Throughput Trend', fontsize=12)
    plt.grid(True, ls='--', alpha=0.6)

    # 子图3：响应时间对比
    plt.subplot(1, 3, 3)
    x = np.arange(len(concs))
    width = 0.25
    plt.bar(x - width, [r['p50'] for r in results], width, label='P50 (Median)', color='#82ccdd')
    plt.bar(x, [r['p95'] for r in results], width, label='P95', color='#60a3bc')
    plt.bar(x + width, [r['p99'] for r in results], width, label='P99 (Worst)', color='#0a3d62')
    plt.xticks(x, concs)
    plt.xlabel('Concurrency')
    plt.ylabel('Latency (ms)')
    plt.title('Latency Percentiles by Load', fontsize=12)
    plt.legend()
    plt.grid(axis='y', ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig("./results/system_test/stress_test_report.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    samples = load_test_data()
    if not samples:
        print("错误: 未能在指定路径找到有效的图片数据，请检查 data/ 目录。")
        exit()

    concurrent_levels = [10, 50, 100]
    all_metrics = []

    for c in concurrent_levels:
        res = run_stress_test(c, samples, num_requests=200)
        if res:
            all_metrics.append(res)

    if all_metrics:
        print("\n" + "=" * 65)
        print(f"{'并发数':<8} | {'QPS':<10} | {'Avg(ms)':<10} | {'P95(ms)':<10} | {'P99(ms)':<10}")
        print("-" * 65)
        for r in all_metrics:
            print(f"{r['concurrency']:<8} | "
                  f"{r['qps']:<10.2f} | "
                  f"{r['avg']:<10.1f} | "
                  f"{r['p95']:<10.1f} | "
                  f"{r['p99']:<10.1f}")
        print("=" * 65)
        visualize_results(all_metrics)
