import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import LABEL_MAPPING


class DataPipeline:
    def __init__(self, filepath):
        self.filepath = filepath
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        print("[*] Loading Dataset...")
        df = pd.read_csv(self.filepath)
        df.columns = df.columns.str.strip()

        if 'Label' not in df.columns:
            raise ValueError("Column 'Label' not found in CSV.")

        df['Target'] = df['Label'].map(LABEL_MAPPING)
        df = df.dropna(subset=['Target'])

        # Feature Engineering
        df = df.dropna()
        nunique = df.nunique()
        cols_to_drop = nunique[nunique <= 1].index
        df = df.drop(cols_to_drop, axis=1)

        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        df = df.drop(to_drop, axis=1)

        print(f"[*] Preprocessing complete. Features reduced to {df.shape[1]-2} (minus Label/Target).")

        X = df.drop(['Label', 'Target'], axis=1)
        y = df['Target']

        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y.values

    def balance_data(self, X, y):
        print("[*] Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res