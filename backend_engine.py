import os
import sqlite3

import faiss
import numpy as np
import yaml

from utils import EnhancedFeatureExtractor

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    CONF = yaml.safe_load(f)


class SearchEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SearchEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized: return
        print(">>> 初始化核心引擎...")

        # 1. 初始化 SQLite 数据库
        self._init_db_schema()

        # 2. 初始化 FAISS
        self.code_length = CONF['model']['hash_code_length']
        self.index = faiss.IndexBinaryFlat(self.code_length)

        # 3. 加载矩阵
        self._load_shoh_matrices()

        # 4. 加载特征提取器
        self.feature_extractor = EnhancedFeatureExtractor()

        # 5. 加载现有索引
        if os.path.exists(CONF['paths']['faiss_index_file']):
            print(">>> 加载现有的 FAISS 索引...")
            self.index = faiss.read_index_binary(CONF['paths']['faiss_index_file'])

        self.initialized = True
        print(">>> 引擎初始化完成。")

    def _get_conn(self):
        """获取新的数据库连接（修复递归游标错误）"""
        return sqlite3.connect(CONF['paths']['database_file'], check_same_thread=False)

    def _init_db_schema(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    title TEXT,
                    brand TEXT,
                    label_raw TEXT
                )
            ''')
            conn.commit()

    def _load_shoh_matrices(self):
        res_dir = CONF['paths']['result_dir']
        try:
            self.Wx = np.load(os.path.join(res_dir, 'Wx_matrix.npy'))
            self.Wy = np.load(os.path.join(res_dir, 'Wy_matrix.npy'))
        except Exception as e:
            print(f"Error loading matrices: {e}")
            self.Wx, self.Wy = None, None

    def _pack_bits(self, float_hash):
        binary = (float_hash > 0).astype(np.uint8)
        return np.packbits(binary, axis=1)

    def add_product(self, image_path, title, brand, label_raw):
        print(f"正在处理商品: {title}")

        # 1. 提取特征
        # 确保提取器返回的是 numpy array。如果提取器报错，请检查 utils.py
        feat_img = self.feature_extractor.extract_image_features_capture([image_path])
        feat_txt = self.feature_extractor.extract_text_features_capture([title])

        # 2. 生成哈希 (确保使用 Wy 矩阵进行文本映射)
        # 确保乘法后的维度正确 (1, hash_length)
        h_vec = np.dot(feat_txt, self.Wy.T)

        # 3. 转换为二进制并打包
        # np.sign(h_vec) > 0 得到布尔数组，转为 uint8
        binary_array = (h_vec > 0).astype(np.uint8)
        # packbits 将 8 个 0/1 压缩为一个字节
        h_bit = np.packbits(binary_array, axis=1)

        # 4. 更新 FAISS 索引
        # 注意：h_bit 必须是二维数组 (1, n)，如果是 (n,) 需要 reshape
        if len(h_bit.shape) == 1:
            h_bit = h_bit.reshape(1, -1)

        self.index.add(h_bit)

        # 5. 保存索引文件
        faiss.write_index_binary(self.index, CONF['paths']['faiss_index_file'])

        # 6. 写入数据库
        safe_label = str(label_raw) if label_raw else ""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO products (image_path, title, brand, label_raw)
                VALUES (?, ?, ?, ?)
            ''', (image_path, title, brand, safe_label))
            conn.commit()
            new_id = cursor.lastrowid

        print(f">>> 商品上架成功 ID: {new_id}")
        return new_id

    def get_stats(self):
        count = 0
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM products')
                # 修复: fetchone 可能返回 None
                row = cursor.fetchone()
                if row:
                    count = row[0]
        except Exception as e:
            print(f"Stats Error: {e}")
        return {"learned_count": count, "model_status": "Active"}

    def search_image(self, image_path, top_k=20):
        feat = self.feature_extractor.extract_image_features_capture([image_path])
        query_hash = np.sign(feat @ self.Wx.T)
        packed = self._pack_bits(query_hash)

        # search 返回两个值: dists, indices
        dists, indices = self.index.search(packed, top_k)
        return self._fetch_results(indices[0], dists[0])

    def search_text(self, text, top_k=20):
        feat = self.feature_extractor.extract_text_features_capture([text])
        query_hash = np.sign(feat @ self.Wy.T)
        packed = self._pack_bits(query_hash)

        # search 返回两个值: dists, indices
        dists, indices = self.index.search(packed, top_k)
        return self._fetch_results(indices[0], dists[0])

    def _fetch_results(self, indices, dists):
        results = []
        valid_indices = [int(i) + 1 for i in indices if i != -1]

        if not valid_indices:
            return []

        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(valid_indices))
                # 假设 ID = index + 1
                query = f'SELECT id, image_path, title, brand, label_raw FROM products WHERE id IN ({placeholders})'
                cursor.execute(query, valid_indices)
                rows = {row[0]: row for row in cursor.fetchall()}
        except Exception as e:
            print(f"Fetch Error: {e}")
            return []

        for i, idx in enumerate(indices):
            if idx == -1: continue
            db_id = int(idx) + 1
            if db_id in rows:
                row = rows[db_id]
                results.append({
                    'image_path': row[1].replace('\\', '/'),
                    'title': row[2],
                    'brand': row[3],
                    'label': row[4],
                    'score': float(dists[i])
                })
        return results

    # 辅助查询接口
    def get_brands(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT brand FROM products')
            return [r[0] for r in cursor.fetchall() if r[0]]

    def get_random_products(self, limit=20):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_path, title, brand, label_raw FROM products ORDER BY RANDOM() LIMIT ?',
                           (limit,))
            return [{
                'image_path': r[0].replace('\\', '/'), 'title': r[1],
                'brand': r[2], 'label': r[3]
            } for r in cursor.fetchall()]

    def filter_by_brand(self, brand, limit=50):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT image_path, title, brand, label_raw FROM products WHERE brand = ? LIMIT ?',
                           (brand, limit))
            return [{
                'image_path': r[0].replace('\\', '/'), 'title': r[1],
                'brand': r[2], 'label': r[3]
            } for r in cursor.fetchall()]


# 单例导出
engine = SearchEngine()
