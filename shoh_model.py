import time

import numpy as np
import torch

from utils import hamming_distance, calculate_map, contrastive_loss


class CAPTUREEnhancedSHOH:
    def __init__(self, params, train, query):
        self.params = params
        self.train = train
        self.query = query
        self.trained_count = 0
        self.train_time = []
        self.B = np.array([])
        self.T = []
        self.Wx = None
        self.Wy = None
        self.C1 = None
        self.C2 = None
        self.cnt1 = None
        self.cnt2 = None

        # CAPTURE parameters
        self.temperature = 0.07
        self.contrastive_weight = 0.3  # Weight for the contrastive loss influence
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")

    def train_shoh(self, chunk_indices, first_round=False):
        tic = time.time()

        # Load parameters
        r = self.params['nbits']
        num_class1 = self.params['num_class1']
        num_class2 = self.params['num_class2']
        dx = self.params['dx']
        dy = self.params['dy']
        n_t = len(chunk_indices)
        eta = self.params['eta']
        alpha1 = self.params['alpha1']
        alpha2 = self.params['alpha2']
        gamma = self.params['gamma']
        xi = self.params['xi']
        mu = self.params['mu']
        max_iter = self.params['max_iter']

        # Load data for the current chunk
        X_new = self.train['X'][chunk_indices, :]
        Y_new = self.train['Y'][chunk_indices, :]
        L1_new = self.train['L1'][chunk_indices, :]
        L2_new = self.train['L2'][chunk_indices, :]

        # Initialization
        if first_round:
            self.B = np.zeros((r, 0))
            self.trained_count = 0
            self.train_time = []
            self.cnt1 = np.zeros(num_class1)
            self.cnt2 = np.zeros(num_class2)

            self.T = [
                np.zeros((r, num_class2)),
                np.zeros((r, num_class1)),
                np.zeros((r, r - 1)),
                np.zeros((r, dx)),
                np.zeros((r, dy)),
                np.zeros((dx, dx)),
                np.zeros((dy, dy)),
                np.zeros((dx, num_class1)),
                np.zeros((dx, num_class2)),
                np.zeros((dy, num_class1)),
                np.zeros((dy, num_class2)),
            ]

            self.C1 = np.sign(np.random.randn(r, num_class1))
            self.C1[self.C1 == 0] = -1
            self.C2 = np.sign(np.random.randn(r, num_class2))
            self.C2[self.C2 == 0] = -1

        # Calculate A1_2
        if 'A1_2' not in self.train:
            self.train['A1_2'] = np.linalg.pinv(self.train['L1'].T @ self.train['L1']) @ self.train['L1'].T @ \
                                 self.train['L2']

        # Construct soft similarity labels
        S1_new = L1_new.T
        V2_new = L1_new @ self.train['A1_2'] + L2_new
        S2_new = np.zeros_like(L2_new)

        for row in range(n_t):
            norm_v = np.linalg.norm(V2_new[row, :])
            if norm_v > 0:
                S2_new[row, :] = V2_new[row, :] / norm_v + gamma * L2_new[row, :]
            else:
                S2_new[row, :] = gamma * L2_new[row, :]
        S2_new = S2_new.T

        # Initialize B_new
        B_new = np.sign(np.random.randn(r, n_t))
        B_new[B_new == 0] = -1

        for iter in range(max_iter):
            P = r * (alpha1 * self.C1 @ S1_new + alpha2 * self.C2 @ S2_new)

            # Add contrastive loss influence
            if self.Wx is not None and self.Wy is not None:
                try:
                    proj_X = torch.FloatTensor(X_new @ self.Wx.T).to(self.device)
                    proj_Y = torch.FloatTensor(Y_new @ self.Wy.T).to(self.device)

                    B_torch = torch.FloatTensor(B_new.T).to(self.device).requires_grad_(True)

                    # Combining aligning projected features and aligning B with those features
                    loss_align_BX = torch.mean(torch.pow(B_torch - proj_X, 2))
                    loss_align_BY = torch.mean(torch.pow(B_torch - proj_Y, 2))
                    loss_contrast = contrastive_loss(proj_X, proj_Y, temperature=self.temperature)

                    total_loss = loss_contrast + 0.1 * (loss_align_BX + loss_align_BY)

                    total_loss.backward()
                    contrast_grad = B_torch.grad.T.cpu().numpy()

                    # Update term P
                    P = P - self.contrastive_weight * contrast_grad

                except Exception as e:
                    print(f"Learning gradient calculation failed: {e}, skipping.")

            for l in range(r):
                idx_exc = np.array(list(range(l)) + list(range(l + 1, r)))
                p_l = P[l, :]
                c1_l = self.C1[l, :]
                C1_ = self.C1[idx_exc, :]
                c2_l = self.C2[l, :]
                C2_ = self.C2[idx_exc, :]
                B_new_ = B_new[idx_exc, :]

                term = (alpha1 * c1_l @ C1_.T + alpha2 * c2_l @ C2_.T) @ B_new_
                b_new_l = np.sign(p_l - term)
                b_new_l[b_new_l == 0] = -1
                B_new[l, :] = b_new_l

            # update C2
            Q2 = r * (alpha2 * (B_new @ S2_new.T + self.T[0]) + eta * self.C1 @ self.train['A1_2'])
            for l in range(r):
                idx_exc = np.array(list(range(l)) + list(range(l + 1, r)))
                q2_l = Q2[l, :]
                b_new_l = B_new[l, :]
                B_new_ = B_new[idx_exc, :]
                c1_l = self.C1[l, :]
                C1_ = self.C1[idx_exc, :]
                C2_ = self.C2[idx_exc, :]

                term = (alpha2 * b_new_l @ B_new_.T + alpha2 * self.T[2][l, :] + eta * c1_l @ C1_.T) @ C2_
                c2_l_new = np.sign(q2_l - term)
                c2_l_new[c2_l_new == 0] = -1
                self.C2[l, :] = c2_l_new

            # update C1
            Q1 = r * (alpha1 * (B_new @ S1_new.T + self.T[1]) + eta * self.C2 @ self.train['A1_2'].T)
            for l in range(r):
                idx_exc = np.array(list(range(l)) + list(range(l + 1, r)))
                q1_l = Q1[l, :]
                b_new_l = B_new[l, :]
                B_new_ = B_new[idx_exc, :]
                C1_ = self.C1[idx_exc, :]
                c2_l = self.C2[l, :]
                C2_ = self.C2[idx_exc, :]

                term = (alpha1 * b_new_l @ B_new_.T + alpha1 * self.T[2][l, :] + eta * c2_l @ C2_.T) @ C1_
                c1_l_new = np.sign(q1_l - term)
                c1_l_new[c1_l_new == 0] = -1
                self.C1[l, :] = c1_l_new

        # Update temporary variables T
        self.B = np.hstack((self.B, B_new))
        self.T[0] += B_new @ S2_new.T
        self.T[1] += B_new @ S1_new.T

        for l in range(r):
            idx_exc = np.array([k for k in range(r) if k != l])
            b_row = B_new[l, :]
            B_exc = B_new[idx_exc, :]
            self.T[2][l, :] += b_row @ B_exc.T

        self.T[3] += B_new @ X_new
        self.T[4] += B_new @ Y_new
        self.T[5] += X_new.T @ X_new
        self.T[6] += Y_new.T @ Y_new

        # Update mean feature vectors
        for i in range(num_class1):
            ddx = np.where(L1_new[:, i] == 1)[0]
            cnt = len(ddx)
            if cnt > 0:
                self.T[7][:, i] = (self.T[7][:, i] * self.cnt1[i] + np.sum(X_new[ddx, :], axis=0)) / (
                        self.cnt1[i] + cnt)
                self.T[9][:, i] = (self.T[9][:, i] * self.cnt1[i] + np.sum(Y_new[ddx, :], axis=0)) / (
                        self.cnt1[i] + cnt)
                self.cnt1[i] += cnt

        for i in range(num_class2):
            ddx = np.where(L2_new[:, i] == 1)[0]
            cnt = len(ddx)
            if cnt > 0:
                self.T[8][:, i] = (self.T[8][:, i] * self.cnt2[i] + np.sum(X_new[ddx, :], axis=0)) / (
                        self.cnt2[i] + cnt)
                self.T[10][:, i] = (self.T[10][:, i] * self.cnt2[i] + np.sum(Y_new[ddx, :], axis=0)) / (
                        self.cnt2[i] + cnt)
                self.cnt2[i] += cnt

        # b in Ax=b
        term1_x = self.T[3] + mu * (alpha1 * self.C1 @ self.T[7].T + alpha2 * self.C2 @ self.T[8].T)
        # A in Ax=b
        term2_x = self.T[5] + mu * (alpha1 * self.T[7] @ self.T[7].T + alpha2 * self.T[8] @ self.T[8].T) + xi * np.eye(
            dx)

        # Regularize Wx and Wy
        if self.Wx is not None and self.Wy is not None:
            alignment_reg = 0.01  # regularization factor
            term2_x += alignment_reg * (X_new.T @ X_new - Y_new.T @ Y_new)

        try:
            # Solve Wx
            self.Wx = np.linalg.solve(term2_x, term1_x.T).T
        except np.linalg.LinAlgError:
            print("Warning: Wx matrix is singular")
            self.Wx = term1_x @ np.linalg.pinv(term2_x)

        term1_y = self.T[4] + mu * (alpha1 * self.C1 @ self.T[9].T + alpha2 * self.C2 @ self.T[10].T)
        term2_y = self.T[6] + mu * (
                alpha1 * self.T[9] @ self.T[9].T + alpha2 * self.T[10] @ self.T[10].T) + xi * np.eye(dy)

        try:
            self.Wy = np.linalg.solve(term2_y, term1_y.T).T
        except np.linalg.LinAlgError:
            print("Warning: Wy matrix is singular, using pseudo-inverse.")
            self.Wy = term1_y @ np.linalg.pinv(term2_y)

        self.trained_count += n_t
        self.train_time.append(time.time() - tic)

        print(f"Training round complete. Total samples trained: {self.trained_count}")

    def evaluate(self, type='standard'):
        if self.Wx is None or self.Wy is None or self.B.size == 0:
            print("Warning: Model has not been trained. Cannot evaluate.")
            return {'map_image2text': 0.0, 'map_text2image': 0.0}

        FxQuery = self.query['X'] @ self.Wx.T
        FyQuery = self.query['Y'] @ self.Wy.T
        B_train = self.B.T

        queryL_GT = self.query['L2']
        trainL_GT = self.train['L2'][:self.trained_count, :]

        eva = {}

        if type == 'standard':
            BxQuery = np.sign(FxQuery)
            ByQuery = np.sign(FyQuery)
            BxTrain = np.sign(B_train)
            ByTrain = BxTrain

            # I->T
            D = hamming_distance(BxQuery, ByTrain)
            idx_rank = np.argsort(D, axis=1).T
            eva['map_image2text'] = calculate_map(idx_rank, trainL_GT, queryL_GT)

            # T->I
            D = hamming_distance(ByQuery, BxTrain)
            idx_rank = np.argsort(D, axis=1).T
            eva['map_text2image'] = calculate_map(idx_rank, trainL_GT, queryL_GT)

        elif type == 'WRS':
            BxTrain = B_train
            ByTrain = B_train
            BxQuery = FxQuery
            ByQuery = FyQuery

            # Using cosine similarity
            epsilon = 1e-8
            norm_BxQuery = np.linalg.norm(BxQuery, axis=1, keepdims=True) + epsilon
            norm_ByTrain = np.linalg.norm(ByTrain, axis=1, keepdims=True) + epsilon
            cos_sim_i2t = (BxQuery @ ByTrain.T) / (norm_BxQuery @ norm_ByTrain.T)
            idx_rank_i2t = np.argsort(-cos_sim_i2t, axis=1).T
            eva['map_image2text'] = calculate_map(idx_rank_i2t, trainL_GT, queryL_GT)

            norm_ByQuery = np.linalg.norm(ByQuery, axis=1, keepdims=True) + epsilon
            norm_BxTrain = np.linalg.norm(BxTrain, axis=1, keepdims=True) + epsilon
            cos_sim_t2i = (ByQuery @ BxTrain.T) / (norm_ByQuery @ norm_BxTrain.T)
            idx_rank_t2i = np.argsort(-cos_sim_t2i, axis=1).T
            eva['map_text2image'] = calculate_map(idx_rank_t2i, trainL_GT, queryL_GT)
        return eva
