import os

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename

from utils import EnhancedFeatureExtractor

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print("Loading pre-trained model and database...")
try:
    # Path to the results directory
    MODEL_DIR = './results/SHOH_CAPTURE_Enhanced/custom_data/'
    # Load the projection matrices
    Wx = np.load(os.path.join(MODEL_DIR, 'Wx_matrix.npy'))
    Wy = np.load(os.path.join(MODEL_DIR, 'Wy_matrix.npy'))

    # Load the entire database of hash codes
    B_database = np.load(os.path.join(MODEL_DIR, 'B_matrix.npy')).T  # Transpose for easier distance calc

    # Load the real database metadata
    db_paths_file = os.path.join(MODEL_DIR, 'db_image_paths.npy')
    print(f"Loading {db_paths_file} ...")
    db_image_paths = np.load(db_paths_file)
    print(f"Successfully loaded {len(db_image_paths)} database image paths.")

    # Load the feature extractor
    feature_extractor = EnhancedFeatureExtractor()

    print("Model and database loaded successfully.")

except FileNotFoundError:
    print("ERROR: Model files not found. Please run the training script first.")
    Wx, Wy, B_database, feature_extractor = None, None, None, None


@app.route('/data/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('data/images', filename)


def hamming_distance(B1, B2):
    return 0.5 * (B2.shape[1] - np.dot(B1, B2.T))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/query/text', methods=['POST'])
def query_by_text():
    top_k = 10

    if Wy is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    query_text = data.get('text', '')
    if not query_text:
        return jsonify({'error': 'No text provided'}), 400
    text_features = feature_extractor.extract_text_features_capture([query_text])
    query_hash = np.sign(text_features @ Wy.T)  # Features to hash code
    distances = hamming_distance(query_hash, B_database)
    ranked_indices = np.argsort(distances.flatten())[:top_k]
    results = [{'image_path': db_image_paths[i]} for i in ranked_indices]

    return jsonify({'results': results})


@app.route('/query/image', methods=['POST'])
def query_by_image():
    top_k = 10

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image_features = feature_extractor.extract_image_features_capture([filepath])
        query_hash = np.sign(image_features @ Wx.T)
        distances = hamming_distance(query_hash, B_database)
        ranked_indices = np.argsort(distances.flatten())[:top_k]
        results = [{'image_path': db_image_paths[i]} for i in ranked_indices]

        return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True)
