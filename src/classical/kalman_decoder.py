import numpy as np
from scipy.linalg import pinv, inv
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
import warnings

class KalmanDecoderBaseline:
    def __init__(self, latent_dim=16, use_pca=True, ridge_alpha=0.0):
        self.latent_dim = latent_dim
        self.use_pca = use_pca
        self.ridge_alpha = ridge_alpha
        self.C = None  # Projection matrix (neural -> latent)
        self.A = None  # Dynamics matrix
        self.Q = None  # Process noise covariance
        self.R = None  # Observation noise covariance
        self.W_out = None  # Linear classifier weights
        self.b_out = None  # Linear classifier bias
        self.symbol_set = None

    def fit(self, neural_data, label_data, symbol_set):
        self.symbol_set = symbol_set
        # Stack all neural data for PCA/projection
        Y = np.concatenate(neural_data, axis=0)  # [total_T x D]
        if self.use_pca:
            # Center and compute PCA
            Y_mean = Y.mean(axis=0)
            Yc = Y - Y_mean
            U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
            self.C = Vt[:self.latent_dim].T  # [D x latent_dim]
        else:
            # Use random projection
            D = Y.shape[1]
            self.C = np.random.randn(D, self.latent_dim)
        # Project all neural data to latent
        X = [y @ self.C for y in neural_data]  # list of [T x latent_dim]
        # Estimate A by regressing x_{t+1} on x_t
        X_all = np.concatenate(X, axis=0)
        X_prev = X_all[:-1]
        X_next = X_all[1:]
        if self.ridge_alpha > 0:
            # Use ridge regression for regularized estimation
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(X_prev, X_next)
            self.A = ridge.coef_.T  # [latent_dim x latent_dim]
        else:
            self.A = pinv(X_prev) @ X_next  # [latent_dim x latent_dim]
        # Estimate Q, R as diagonal covariances
        Q_resid = X_next - X_prev @ self.A
        self.Q = np.diag(np.var(Q_resid, axis=0) + 1e-6)
        Y_proj = [x @ self.C.T for x in X]  # reconstruct neural
        R_resid = np.concatenate(neural_data, axis=0) - np.concatenate(Y_proj, axis=0)
        self.R = np.diag(np.var(R_resid, axis=0) + 1e-6)
        # Prepare classifier targets (framewise, aligned)
        X_frames = []
        Y_labels = []
        for x_seq, y_seq in zip(X, label_data):
            T = min(x_seq.shape[0], y_seq.shape[0])
            X_frames.append(x_seq[:T])
            Y_labels.append(y_seq[:T])
        X_frames = np.concatenate(X_frames, axis=0)
        Y_labels = np.concatenate(Y_labels, axis=0)
        n_symbols = len(symbol_set)
        # One-hot targets
        Y_onehot = np.zeros((Y_labels.shape[0], n_symbols))
        Y_onehot[np.arange(Y_labels.shape[0]), Y_labels] = 1
        # Linear regression for classifier
        if self.ridge_alpha > 0:
            # Use ridge regression for regularized classifier
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            ridge.fit(X_frames, Y_onehot)
            self.W_out = ridge.coef_.T  # [latent_dim x n_symbols]
            self.b_out = Y_onehot.mean(axis=0)
        else:
            self.W_out = pinv(X_frames) @ Y_onehot  # [latent_dim x n_symbols]
            self.b_out = Y_onehot.mean(axis=0)

    def kalman_filter(self, y_seq):
        T, D = y_seq.shape
        x_hat = np.zeros((T, self.latent_dim))
        P = np.eye(self.latent_dim)
        x = np.zeros(self.latent_dim)
        for t in range(T):
            # Predict
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.Q
            # Update
            y = y_seq[t]
            K = P_pred @ self.C.T @ pinv(self.C @ P_pred @ self.C.T + self.R)
            x = x_pred + K @ (y - self.C @ x_pred)
            P = (np.eye(self.latent_dim) - K @ self.C) @ P_pred
            x_hat[t] = x
        return x_hat

    def decode_batch(self, neural_batch):
        logits_batch = []
        for y_seq in neural_batch:
            x_hat = self.kalman_filter(y_seq)
            logits = x_hat @ self.W_out + self.b_out  # [T x n_symbols]
            logits_batch.append(logits)
        return logits_batch

    def predict(self, neural_batch):
        logits_batch = self.decode_batch(neural_batch)
        pred_seqs = [np.argmax(logits, axis=-1) for logits in logits_batch]
        return pred_seqs
