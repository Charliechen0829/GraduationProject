import os

import yaml
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# 引入核心检索引擎 (封装了 FAISS, SQLite, 模型推理)
# 确保项目目录下有 backend_engine.py
from backend_engine import engine

app = Flask(__name__)

# --- 1. 加载配置 ---
# 建议将所有路径放入 config.yaml 管理，这里为了代码独立性保留默认值逻辑
if os.path.exists('config.yaml'):
    with open('config.yaml', 'r', encoding='utf-8') as f:
        CONF = yaml.safe_load(f)
else:
    # 默认配置兜底
    CONF = {
        'paths': {
            'upload_folder': './uploads',
            'data_dir': './data/',
            'database_file': './data/metadata.db'
        },
        'server': {
            'host': '0.0.0.0',
            'port': 5000,
            'admin_token': 'SECRET_ADMIN_123'
        }
    }

app.config['UPLOAD_FOLDER'] = CONF['paths']['upload_folder']
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 2. 初始化引擎 ---
# 这步会加载 ViT/BERT 模型、连接 SQLite、加载 FAISS 索引
print("正在初始化检索引擎...")
engine.initialize()
print("检索引擎就绪。")


# --- 3. 静态资源路由 ---

@app.route('/data/images/<path:filename>')
def serve_images(filename):
    """映射本地图片目录到 Web URL"""
    return send_from_directory(os.path.join(CONF['paths']['data_dir'], 'images'), filename)


@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """映射用户上传的搜索图"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# --- 4. 页面路由 ---

@app.route('/')
def index():
    return render_template('index.html')


# --- 5. 核心业务接口 ---

@app.route('/api/init_data', methods=['GET'])
def init_data():
    """
    首页初始化：
    1. 随机展示 20 个商品 (冷启动)
    2. 返回所有品牌列表供侧边栏筛选
    """
    try:
        results = engine.get_random_products(20)
        brands = engine.get_brands()
        return jsonify({
            'results': results,
            'brands': sorted(brands)
        })
    except Exception as e:
        print(f"Init Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter_by_brand', methods=['POST'])
def filter_by_brand():
    """
    品牌筛选接口 (硬过滤)
    直接查 SQLite 数据库
    """
    data = request.get_json()
    brand_name = data.get('brand', '')
    results = engine.filter_by_brand(brand_name)
    return jsonify({'results': results})


# @app.route('/query/text_base', methods=['POST'])
# def query_by_text_base():
#     """
#     【方案 A: Base】纯向量检索逻辑：
#     1. Vector Recall: 利用语义哈希从 FAISS 直接召回 Top 20。
#     2. 无任何文本过滤或二次排序，直接返回模型原始预测结果。
#     """
#     top_k_display = 20
#
#     data = request.get_json()
#     query_text = data.get('text', '').strip().lower()
#
#     if not query_text:
#         return jsonify({'error': '请输入搜索关键词'}), 400
#
#     # 直接调用引擎进行向量检索
#     # 方案 A 只需要请求最终展示的数量即可
#     final_results = engine.search_text(query_text, top_k=top_k_display)
#
#     print(f"[Ablation A] 搜索词: {query_text} | 原始向量召回: {len(final_results)}")
#
#     return jsonify({
#         'results': final_results,
#         'match_info': {
#             'mode': 'Vector Only (Base)',
#             'count': len(final_results)
#         }
#     })


@app.route('/query/text', methods=['POST'])
def query_by_text():
    """
    【核心修改逻辑】混合检索策略：
    1. Vector Recall (向量召回): 利用哈希语义从 FAISS 召回 Top 200。
    2. Text Filter (文本过滤): 在这 200 个结果中，优先筛选 Title 包含关键词的。
    3. Rerank (重排序):
       - 第一梯队: 包含关键词的商品 (按哈希距离排序)
       - 第二梯队: 不包含关键词但语义接近的商品 (按哈希距离排序)
    """
    # 最终展示数量
    top_k_display = 20
    # 扩大候选池 (候选池越大，包含关键词的概率越高，但性能开销略增)
    candidate_pool_size = 500

    data = request.get_json()
    query_text = data.get('text', '').strip().lower()

    if not query_text:
        return jsonify({'error': '请输入搜索关键词'}), 400

    # 1. 向量召回 (Semantic Search)
    # 调用引擎底层，获取较多的候选集
    candidates = engine.search_text(query_text, top_k=candidate_pool_size)

    # 2. 内存重排序 (In-Memory Reranking)
    exact_matches = []
    semantic_matches = []

    # 遍历候选集进行分类
    for item in candidates:
        # 获取商品标题并转小写
        item_title = item.get('title', '').lower()

        # 判定逻辑：搜索词是否出现在标题中
        if query_text in item_title:
            exact_matches.append(item)
        else:
            semantic_matches.append(item)

    # 3. 组合结果
    # 优先展示精确匹配，如果不足 20 个，用语义匹配补齐
    final_results = (exact_matches + semantic_matches)[:top_k_display]

    print(f"搜索词: {query_text} | 召回: {len(candidates)} | 精确匹配: {len(exact_matches)}")

    return jsonify({
        'results': final_results,
        'match_info': {
            'exact_count': len(exact_matches),
            'semantic_fallback': len(semantic_matches)
        }
    })


@app.route('/query/image', methods=['POST'])
def query_by_image():
    """
    以图搜图接口
    策略: 纯语义哈希检索 (Visual Similarity)
    """
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    # 保存上传的图片
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 调用引擎进行检索
    results = engine.search_image(filepath, top_k=20)

    return jsonify({'results': results})


# --- 6. 管理员接口 (动态更新) ---

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """返回模型演进状态"""
    return jsonify(engine.get_stats())


@app.route('/api/admin/add_product', methods=['POST'])
def add_product():
    """管理员增量上架接口"""
    # 权限检查 (简单 Demo 使用 Token)
    if request.headers.get('Authorization') != CONF['server']['admin_token']:
        return jsonify({'error': 'Unauthorized'}), 403

    img_file = request.files.get('image')
    title = request.form.get('title')
    brand = request.form.get('brand', 'Other')
    label = request.form.get('label', '')

    if not img_file or not title:
        return jsonify({'error': 'Missing file or title'}), 400

    # 保存图片到指定目录
    filename = secure_filename(img_file.filename)
    rel_path = f"images/{filename}"
    abs_path = os.path.join(CONF['paths']['data_dir'], rel_path)
    img_file.save(abs_path)

    try:
        # 调用引擎增量处理
        new_id = engine.add_product(abs_path, title, brand, label)
        return jsonify({'success': True, 'id': new_id, 'message': 'Successfully learned new product'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 启动服务
    app.run(
        host=CONF['server']['host'],
        port=CONF['server']['port'],
        debug=True
    )
