import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, ViTModel, BertModel


class EnhancedFeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")

        try:
            # image feature extractor - ViT
            self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
            self.vit_model.eval()
        except Exception as e:
            print(f"ViT model loading failed: {e}, will use CLIP instead")
            self.vit_model = None

        try:
            # text feature extractor -  BERT
            self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
            self.bert_model.eval()
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception as e:
            print(f"BERT model loading failed: {e}, will use CLIP instead")
            self.bert_model = None
            self.bert_tokenizer = None

        try:
            # CLIP as fallback
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_model.eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"CLIP model loading failed: {e}")
            self.clip_model = None
            self.clip_processor = None

    def extract_image_features_capture(self, image_paths):
        features = []
        for image_path in tqdm(image_paths, desc="Extracting image features"):
            try:
                image = Image.open(image_path).convert("RGB")
                combined_features = []

                # using ViT to extract deep features
                if self.vit_model is not None and self.clip_processor is not None:
                    try:
                        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            vit_outputs = self.vit_model(inputs.pixel_values)
                            image_features = vit_outputs.last_hidden_state.mean(dim=1)
                            combined_features.append(image_features)
                    except Exception as e:
                        print(f"ViT feature extraction failed: {e}")

                # using CLIP as fallback
                if self.clip_model is not None and self.clip_processor is not None:
                    try:
                        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            clip_features = self.clip_model.get_image_features(**inputs)
                            combined_features.append(clip_features)
                    except Exception as e:
                        print(f"CLIP image feature extraction failed: {e}")

                if combined_features:
                    # feature fusion
                    final_features = torch.cat(combined_features, dim=1)
                    features.append(final_features.squeeze(0).cpu().numpy())
                else:
                    # fallback feature
                    features.append(np.zeros(512))

            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                features.append(np.zeros(512))

        return np.array(features)

    # using BERT and CLIP to extract text features
    def extract_text_features_capture(self, texts):
        features = []
        for text in tqdm(texts, desc="Extracting text features"):
            try:
                combined_features = []

                # using BERT to extract deep text features
                if self.bert_model is not None and self.bert_tokenizer is not None:
                    try:
                        inputs = self.bert_tokenizer(text, return_tensors="pt",
                                                     padding=True, truncation=True,
                                                     max_length=128).to(self.device)
                        with torch.no_grad():
                            bert_outputs = self.bert_model(**inputs)
                            text_features = bert_outputs.last_hidden_state.mean(dim=1)
                            combined_features.append(text_features)
                    except Exception as e:
                        print(f"BERT特征提取失败: {e}")

                # using CLIP as fallback
                if self.clip_model is not None and self.clip_processor is not None:
                    try:
                        clip_inputs = self.clip_processor(text=text, return_tensors="pt",
                                                          padding=True, truncation=True,
                                                          max_length=77).to(self.device)
                        with torch.no_grad():
                            clip_text_features = self.clip_model.get_text_features(**clip_inputs)
                            combined_features.append(clip_text_features)
                    except Exception as e:
                        print(f"CLIP文本特征提取失败: {e}")

                if combined_features:
                    # feature fusion
                    final_features = torch.cat(combined_features, dim=1)
                    features.append(final_features.squeeze(0).cpu().numpy())
                else:
                    # fallback feature
                    features.append(np.zeros(512))

            except Exception as e:
                print(f"Error processing text {text}: {e}")
                features.append(np.zeros(512))

        return np.array(features)


# init
enhanced_extractor = EnhancedFeatureExtractor()


def extract_image_features(image_paths):
    return enhanced_extractor.extract_image_features_capture(image_paths)


def extract_text_features(texts):
    return enhanced_extractor.extract_text_features_capture(texts)


def contrastive_loss(features1, features2, temperature=0.07):
    # normalize features
    features1 = torch.nn.functional.normalize(features1, dim=1)
    features2 = torch.nn.functional.normalize(features2, dim=1)

    # compute similarity matrix
    similarity_matrix = torch.matmul(features1, features2.T) / temperature

    # create labels (diagonal elements are positive samples)
    batch_size = features1.shape[0]
    labels = torch.arange(batch_size).to(features1.device)

    # compute cross-entropy loss
    loss_i2t = torch.nn.functional.cross_entropy(similarity_matrix, labels, label_smoothing=0.1)
    loss_t2i = torch.nn.functional.cross_entropy(similarity_matrix.T, labels, label_smoothing=0.1)

    return (loss_i2t + loss_t2i) / 2


# Evaluation Metrics
def hamming_distance(B1, B2):
    """
    Computes Hamming distance
    B1: (n_query, n_bits), {-1, 1}
    B2: (n_train, n_bits), {-1, 1}
    Returns: (n_query, n_train) distance matrix.
    """
    # The formula 0.5 * (nbits - B1 @ B2.T) is a fast way to compute
    # Hamming distance for {-1, 1} encoded bits.
    return 0.5 * (B2.shape[1] - B1 @ B2.T)


def compact_bit(B):
    """
    This function is not strictly needed if using the {-1, 1} hamming_distance,
    but we keep it for conceptual clarity. It binarizes the codes.
    """
    return np.sign(B)


def calculate_map(idx_rank, trainL_GT, queryL_GT):
    """
    Calculate mAP.
    idx_rank: (n_train, n_query) - ranked list of training indices for each query
    trainL_GT: (n_train, n_labels)
    queryL_GT: (n_query, n_labels)
    """
    nquery = idx_rank.shape[1]
    APx = np.zeros(nquery)
    # R is the number of items to consider in the ranked list, here we use all.
    R = trainL_GT.shape[0]

    for i in range(nquery):
        label = queryL_GT[i, :]
        # Get the ranked labels
        ranked_labels = trainL_GT[idx_rank[:, i], :]
        # Check for matches (at least one common label)
        imatch = (ranked_labels @ label.T) > 0

        imatch_all = np.sum(imatch)
        if imatch_all == 0:
            continue

        Lx = np.cumsum(imatch)
        Px = Lx / np.arange(1, R + 1)
        APx[i] = np.sum(Px * imatch) / imatch_all

    return np.mean(APx)


def calculate_precision_recall(idx_rank, trainL_GT, queryL_GT):
    """
    Calculate precision and recall curves.
    """
    nquery = idx_rank.shape[1]
    K = idx_rank.shape[0]  # Number of retrieval items
    P = np.zeros((K, nquery))
    R = np.zeros((K, nquery))

    for i in range(nquery):
        label = queryL_GT[i, :]
        ranked_labels = trainL_GT[idx_rank[:, i], :]
        imatch = (ranked_labels @ label.T) > 0

        LK = np.sum(imatch)  # Total relevant items for this query
        if LK == 0:
            continue

        Lk = np.cumsum(imatch)
        P[:, i] = Lk / np.arange(1, K + 1)
        R[:, i] = Lk / LK

    # Average over all queries
    mP = np.mean(P, axis=1)
    mR = np.mean(R, axis=1)

    return mP, mR


# Dataset Loading
def load_dataset(params):
    data_dir = params['ds_dir']
    json_file = os.path.join(data_dir, 'train_info.json')
    feature_cache_path = os.path.join(data_dir, 'custom_data_features.npz')

    print("Loading JSON file to get image paths...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items, l1_set, l2_set = [], set(), set()
    for key, value in data.items():
        full_image_path = os.path.join(value['local_path']).replace("\\", "/")

        item = {'id': key, 'image_path': full_image_path, 'title': value['title'], 'L1': [], 'L2': []}
        labels_raw = value.get('label', '')
        for label_segment in labels_raw.split(';'):
            if not label_segment.strip(): continue
            parts = label_segment.strip().split(' ', 1)
            l1_brand = parts[0]
            l2_product = parts[1] if len(parts) > 1 else l1_brand
            item['L1'].append(l1_brand);
            l1_set.add(l1_brand)
            item['L2'].append(l2_product);
            l2_set.add(l2_product)
        if not item['L1']: item['L1'].append('other'); l1_set.add('other')
        if not item['L2']: item['L2'].append('other'); l2_set.add('other')
        items.append(item)

    if os.path.exists(feature_cache_path):
        print("Loading feature cache...")
        data = np.load(feature_cache_path, allow_pickle=True)
        X, Y, L1, L2 = data['X'], data['Y'], data['L1'], data['L2']
    else:
        print("Feature cache not found, extracting features from dataset...")
        X = extract_image_features([item['image_path'] for item in items])
        Y = extract_text_features([item['title'] for item in items])

        l1_list, l2_list = sorted(list(l1_set)), sorted(list(l2_set))
        l1_to_idx, l2_to_idx = {lbl: i for i, lbl in enumerate(l1_list)}, {lbl: i for i, lbl in enumerate(l2_list)}
        L1 = np.zeros((len(items), len(l1_list)))
        L2 = np.zeros((len(items), len(l2_list)))
        for i, item in enumerate(items):
            for lbl in item['L1']: L1[i, l1_to_idx[lbl]] = 1
            for lbl in item['L2']: L2[i, l2_to_idx[lbl]] = 1

        np.savez(feature_cache_path, X=X, Y=Y, L1=L1, L2=L2)
        print(f"Features have been saved to {feature_cache_path}")

    # Set dynamic parameters
    params['num_class1'], params['num_class2'] = L1.shape[1], L2.shape[1]
    params['dx'], params['dy'] = X.shape[1], Y.shape[1]

    # Split dataset
    N = X.shape[0]
    params['nquery'] = max(100, int(N * 0.1))
    params['chunk_size'] = 2000

    np.random.seed(int(time.time()))
    R = np.random.permutation(N)
    iquery, itrain = R[:params['nquery']], R[params['nquery']:]

    query = {'X': X[iquery, :], 'Y': Y[iquery, :], 'L1': L1[iquery, :], 'L2': L2[iquery, :], 'size': len(iquery)}
    train = {'X': X[itrain, :], 'Y': Y[itrain, :], 'L1': L1[itrain, :], 'L2': L2[itrain, :], 'size': len(itrain)}

    all_image_paths = [item['image_path'] for item in items]
    train_image_paths = [all_image_paths[i] for i in itrain]

    model_dir = os.path.join(params['rec_dir'], 'SHOH_CAPTURE_Enhanced', params['ds_name'])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.save(os.path.join(model_dir, 'db_image_paths.npy'), train_image_paths)
    print(f"Training set image paths have been saved to {model_dir}")

    return params, train, query
