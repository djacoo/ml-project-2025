import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, MultiLabelBinarizer

CATEGORICAL_FEATURES = [
    "countries",
    "pnns_groups_1",
    "pnns_groups_2"
]

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Feature encoder for categorical features
    
    Encoding strategies:
    - countries: MultiLabelBinarizer (top N countries, configurable)
    - pnns_groups_1: OneHotEncoder
    - pnns_groups_2: TargetEncoder
    
    Parameters:
    -----------
    top_n_countries : int, default=15
        Number of top countries to keep for MultiLabelBinarizer encoding
    """
    def __init__(self, top_n_countries=15):
        self.top_n_countries = top_n_countries
        self.encoders = {}
        self.top_countries = None  # Store top countries for countries feature
    
    def fit(self, X, y=None):
        # reset encoders dictionary and top_countries
        self.encoders = {}
        self.top_countries = None
        
        for feature in CATEGORICAL_FEATURES:
            if feature not in X.columns:
                continue  
            
            # Prepare data and configure encoder based on feature type
            if feature == "countries":
                # Convert comma-separated strings to lists of countries
                data = X[feature].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip() and c.strip() != 'unknown']).tolist()
                
                # Count frequency of each country to find top N
                from collections import Counter
                country_counts = Counter()
                for country_list in data:
                    country_counts.update(country_list)
                
                # Get top N countries
                top_countries = [country for country, _ in country_counts.most_common(self.top_n_countries)]
                self.top_countries = top_countries
                
                # Use MultiLabelBinarizer with only top N countries
                encoder = MultiLabelBinarizer(classes=top_countries)
                # Filter data to only include top countries
                filtered_data = [[c for c in country_list if c in self.top_countries] for country_list in data]
                encoder.fit(filtered_data)
                self.encoders[feature] = encoder
            elif feature == "pnns_groups_1":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                data = X[feature].values.reshape(-1, 1)
                encoder.fit(data)
                self.encoders[feature] = encoder
            elif feature == "pnns_groups_2":
                encoder = TargetEncoder(smooth="auto")
                data = X[feature].values.reshape(-1, 1)
                if y is not None:
                    encoder.fit(data, y)
                    self.encoders[feature] = encoder
                else:
                    raise ValueError("TargetEncoder for "+ feature + " requires y parameter")
        
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
            
            # Process based on encoder type
            if isinstance(encoder, MultiLabelBinarizer):
                # Convert comma-separated strings to lists of countries
                data = X_encoded[feature].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip() and c.strip() != 'unknown']).tolist()
                # Filter to only include top countries (others are ignored)
                filtered_data = [[c for c in country_list if c in self.top_countries] for country_list in data]
                transformed_data = encoder.transform(filtered_data)
                
                # Remove original column and add encoded columns
                X_encoded = X_encoded.drop(columns=[feature])
                cols = [f"{feature}_{country}" for country in encoder.classes]
                df_mlb = pd.DataFrame(transformed_data, columns=cols, index=X_encoded.index)
                X_encoded = pd.concat([X_encoded, df_mlb], axis=1)
            
            elif isinstance(encoder, OneHotEncoder):
                data = X_encoded[feature].values.reshape(-1, 1)
                transformed_data = encoder.transform(data)
                
                # Remove original column and add encoded columns
                X_encoded = X_encoded.drop(columns=[feature])
                cols = encoder.get_feature_names_out([feature])
                df_ohe = pd.DataFrame(transformed_data, columns=cols, index=X_encoded.index)
                X_encoded = pd.concat([X_encoded, df_ohe], axis=1)
            
            else:  # TargetEncoder
                data = X_encoded[feature].values.reshape(-1, 1)
                transformed_data = encoder.transform(data)
                
                # TargetEncoder outputs a single column - ensure 1D array
                transformed_data = np.asarray(transformed_data).squeeze()
                if transformed_data.ndim > 1:
                    transformed_data = transformed_data[:, 0]
                
                # Replace original column with transformed values
                X_encoded[feature] = transformed_data

        return X_encoded

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)