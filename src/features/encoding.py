import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

CATEGORICAL_FEATURES = [
    "countries",
    "pnns_groups_1",
    "pnns_groups_2"
]

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Feature encoder for categorical features
    Uses OneHotEncoder for countries and pnns_groups_1
    Uses TargetEncoder for pnns_groups_2


    """
    def __init__(self):
        self.encoders = {}
    
    def fit(self, X, y=None):
        # reset encoders dictionary
        self.encoders = {}
        
        for feature in CATEGORICAL_FEATURES:
            if feature not in X.columns:
                continue  
            
            # configure encoder based on feature type
            if feature == "countries":
                encoder = OneHotEncoder(max_categories=11, sparse_output=False, handle_unknown="ignore")
            elif feature == "pnns_groups_1":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            elif feature == "pnns_groups_2":
                encoder = TargetEncoder(smooth="auto")
            else:
                continue  # skip unknown features
            
            # reshape to 2D array (N_samples, 1)
            data = X[feature].values.reshape(-1, 1)

            if isinstance(encoder, TargetEncoder):
                if y is not None:
                    encoder.fit(data, y)
                    self.encoders[feature] = encoder
                else:
                    raise ValueError("TargetEncoder for "+ feature + " requires y parameter")
            else:
                encoder.fit(data)
                self.encoders[feature] = encoder
        
        return self

    def transform(self, X):
        X_encoded = X.copy()
        
        for feature in CATEGORICAL_FEATURES:
            # Check if feature exists and encoder is fitted
            if feature not in X_encoded.columns:
                continue  # Skip if feature doesn't exist in transform data
            
            if feature not in self.encoders:
                continue  # Skip if encoder wasn't fitted
            
            encoder = self.encoders[feature]
            
            # Reshape input to 2D for sklearn
            data = X_encoded[feature].values.reshape(-1, 1)
            transformed_data = encoder.transform(data)
            
            # remove original column before adding encoded columns
            X_encoded = X_encoded.drop(columns=[feature])
            
            if isinstance(encoder, OneHotEncoder):
                cols = encoder.get_feature_names_out([feature])
                
                df_ohe = pd.DataFrame(transformed_data, columns=cols, index=X_encoded.index)
                X_encoded = pd.concat([X_encoded, df_ohe], axis=1)
            
            else: 
                # TargetEncoder outputs a single column, replace original feature
                X_encoded[feature] = transformed_data.flatten()

        return X_encoded

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)