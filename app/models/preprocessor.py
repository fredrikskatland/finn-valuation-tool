import pandas as pd
import numpy as np

class InferencePreprocessor:
    def __init__(self, categorical_features, column_order_path):
        self.categorical_features = categorical_features
        self.column_order = pd.read_csv(column_order_path, header=None).iloc[:, 0].tolist()

    def preprocess(self, features_dict):
        """
        Preprocess a single prediction request to align with training columns.
        """
        df = pd.DataFrame([features_dict])
        for feature in self.categorical_features:
            dummies = pd.get_dummies(df[feature], prefix=feature, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[feature], inplace=True)

        # Align columns to match training order, filling missing columns with 0
        processed_df = df.reindex(columns=self.column_order, fill_value=0)
        return processed_df.to_numpy(dtype=np.float32)
