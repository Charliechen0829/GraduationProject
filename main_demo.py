import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from shoh_model import SHOH
from utils import load_dataset


def main():
    data_dir = './data/'
    rec_dir = './results/'

    ds_name_list = ['custom_data']
    nbits_list = [256]
    test_times = 1

    for t in range(1, test_times + 1):
        for ds_name in ds_name_list:
            for nbits in nbits_list:
                print(f'LOAD DATASET: {ds_name}')
                print(f'CODE LENGTH: {nbits}')

                param = {
                    't': t,
                    'ds_name': ds_name,
                    'nbits': nbits,
                    'ds_dir': data_dir,
                    'rec_dir': rec_dir,
                    'alpha1': 0.1,
                    'alpha2': 0.3,
                    'eta': 0.6,
                    'gamma': 0.5,
                    'xi': 0.001,
                    'mu': 800,
                    'max_iter': 10
                }

                # Load dataset
                param, train, query = load_dataset(param)
                print(f"Dataset loaded. Training size: {train['size']}, Query size: {query['size']}")
                print(f"Image feature dim: {param['dx']}, Text feature dim: {param['dy']}")

                print('----------------------- Train SHOH -----------------------')
                shoh_model = SHOH(param, train, query)

                # Chunk-wise training
                # use a smaller chunk size for more fast evaluation
                param['chunk_size'] = 1000
                n_chunks = int(np.ceil(train['size'] / param['chunk_size']))
                first_round = True

                # save the performance
                map_history_i2t = []
                map_history_t2i = []
                map_history_wrs_i2t = []
                map_history_wrs_t2i = []
                trained_samples_history = []

                start_time = time.time()
                for i in range(n_chunks):
                    idx_strt = i * param['chunk_size']
                    idx_end = min(idx_strt + param['chunk_size'], train['size'])
                    chunk_indices = np.arange(idx_strt, idx_end)

                    if len(chunk_indices) == 0:
                        continue

                    print(f'-------------- Round / Total: {i + 1} / {n_chunks} --------------')
                    shoh_model.train_shoh(chunk_indices, first_round=first_round)
                    first_round = False

                    # evaluate
                    eva_standard = shoh_model.evaluate(type='standard')
                    map_history_i2t.append(eva_standard['map_image2text'])
                    map_history_t2i.append(eva_standard['map_text2image'])
                    print(f"MAP of SHOH (I->T): {eva_standard['map_image2text']:.4f}")
                    print(f"MAP of SHOH (T->I): {eva_standard['map_text2image']:.4f}")

                    eva_wrs = shoh_model.evaluate(type='WRS')
                    map_history_wrs_i2t.append(eva_wrs['map_image2text'])
                    map_history_wrs_t2i.append(eva_wrs['map_text2image'])
                    print(f"MAP of SHOH-WRS (I->T): {eva_wrs['map_image2text']:.4f}")
                    print(f"MAP of SHOH-WRS (T->I): {eva_wrs['map_text2image']:.4f}")

                    trained_samples_history.append(shoh_model.trained_count)

                end_time = time.time()
                print(f"Total training time: {end_time - start_time:.4f} seconds")

                print('----------------------- Done -----------------------')

                # visualization
                record_dir = os.path.join(rec_dir, 'SHOH_CAPTURE_Enhanced', param['ds_name'])
                if not os.path.exists(record_dir):
                    os.makedirs(record_dir)

                plt.figure(figsize=(12, 8))
                plt.plot(trained_samples_history, map_history_i2t, 'o-', label='mAP I->T (Standard)')
                plt.plot(trained_samples_history, map_history_t2i, 'o-', label='mAP T->I (Standard)')
                plt.plot(trained_samples_history, map_history_wrs_i2t, 's--', label='mAP I->T (WRS/Cosine)')
                plt.plot(trained_samples_history, map_history_wrs_t2i, 's--', label='mAP T->I (WRS/Cosine)')
                plt.xlabel('Number of Trained Samples')
                plt.ylabel('Mean Average Precision (mAP)')
                plt.title(f'mAP vs. Training Progress ({nbits}-bits)')
                plt.legend()
                plt.grid(True)
                plt.ylim(bottom=0)  # mAP cannot be negative
                plot_path = os.path.join(record_dir, f"map_history_{nbits}bits.png")
                plt.savefig(plot_path)
                plt.show()
                print(f"Performance plot saved to {plot_path}")

                # Saving results
                record_name = f"CAPTURE_Enhanced_test{param['t']}_Bits{param['nbits']}.json"
                result = {
                    'param': param,
                    'map_history_i2t': map_history_i2t,
                    'map_history_t2i': map_history_t2i,
                    'map_history_wrs_i2t': map_history_wrs_i2t,
                    'map_history_wrs_t2i': map_history_wrs_t2i,
                    'train_time': shoh_model.train_time,
                    'model_type': 'SHOH_with_CAPTURE_enhancement'
                }

                # Custom converter for numpy types
                def convert(o):
                    if isinstance(o, (np.int64, np.int32)): return int(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    raise TypeError

                with open(os.path.join(record_dir, record_name), 'w') as f:
                    json.dump(result, f, indent=4, default=convert)

                np.save(os.path.join(record_dir, 'B_matrix.npy'), shoh_model.B)
                np.save(os.path.join(record_dir, 'Wx_matrix.npy'), shoh_model.Wx)
                np.save(os.path.join(record_dir, 'Wy_matrix.npy'), shoh_model.Wy)

                print(f"Results saved to {os.path.join(record_dir, record_name)}")


if __name__ == '__main__':
    main()
