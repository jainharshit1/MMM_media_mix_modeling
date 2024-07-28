import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,ElasticNet,HuberRegressor
from ChannelContrib.carryover import ExponentialCarryover, GaussianCarryover
from ChannelContrib.saturation import ExponentialSaturation, BoxCoxSaturation, HillSaturation
from sklearn.preprocessing import MinMaxScaler

class ExtractKeyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a dictionary and extract the value for the specified key
        return X[self.key]


class PrintTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, step_name):
        self.step_name = step_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"After {self.step_name} transformation:\n", X)
        return X
class NaNChecker(BaseEstimator, TransformerMixin):
    def __init__(self, step_name=''):
        self.step_name = step_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            # Check for NaNs in a NumPy array
            if np.isnan(X).any():
                print(f"NaN values found after {self.step_name}")
            else:
               pass
        elif isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            # Check for NaNs in a pandas DataFrame or Series
            if X.isnull().values.any():
                print(f"NaN values found after {self.step_name}")
            else:
                pass
        elif isinstance(X, dict):
            # Check for NaNs in a dictionary
            for key, value in X.items():
                if isinstance(value, np.ndarray):
                    if np.isnan(value).any():
                        print(f"NaN values found in key '{key}' after {self.step_name}")
                elif isinstance(value, (pd.DataFrame, pd.Series)):
                    if value.isnull().values.any():
                        print(f"NaN values found in key '{key}' after {self.step_name}")
                elif isinstance(value, list):
                    if any(pd.isnull(item) for item in value):
                        print(f"NaN values found in key '{key}' after {self.step_name}")
        else:
            # For other types, we assume no NaN check is necessary
            print(f"Data type {type(X)} in {self.step_name} does not support NaN check")
        return X


class TransformationPipeline_1:
    def get_carryover_instance(self, carryover_model, carryover_params):
        if carryover_model == 'ExponentialCarryover':
            return ExponentialCarryover().set_params(**carryover_params)
        else:
            return GaussianCarryover().set_params(**carryover_params)

    def get_saturation_instance(self, saturation_model, saturation_params):
        if saturation_model == 'ExponentialSaturation':
            return ExponentialSaturation().set_params(**saturation_params)
        elif saturation_model == 'BoxCoxSaturation':
            return BoxCoxSaturation().set_params(**saturation_params)
        # elif saturation_model == 'AdbudgSaturation':
        #     return AdbudgSaturation().set_params(**saturation_params)
        else:
            return HillSaturation()

    def column_transformer(self,explanatory_vars, carryover_model, saturation_model, carryover_params, saturation_params,carryover_key):
        carryover = self.get_carryover_instance(carryover_model, carryover_params)
        saturation = self.get_saturation_instance(saturation_model, saturation_params)
        transformers = []

        for col in explanatory_vars:
            if col in explanatory_vars[:-3]:  # Exclude last 3 columns
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    #
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('carryover', carryover),
                    ('extract_key', ExtractKeyTransformer(carryover_key)),  # Specify the key here
                    ('saturation', saturation),
                     ])
            else:
                pipeline = Pipeline([
                    ('scaler', MinMaxScaler()),
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
                ])
            transformers.append((col, pipeline, [col]))

        adstock = ColumnTransformer(transformers)
        return adstock

    # def column_transformer_1(self,explanatory_vars, carryover_model, saturation_model, carryover_params, saturation_params,carryover_key):
    #     carryover = self.get_carryover_instance(carryover_model, carryover_params)
    #     saturation = self.get_saturation_instance(saturation_model, saturation_params)
    #     transformers = []
    #
    #     for col in explanatory_vars:
    #         if col in explanatory_vars[:-3]:  # Exclude last 3 columns
    #             pipeline = Pipeline([
    #                 ('scaler', Pipeline([
    #                     ('minmax', MinMaxScaler()),
    #                     ('print', PrintTransformer('scaling'))
    #                 ])),
    #                 ('imputer', Pipeline([
    #                     ('simple_imputer', SimpleImputer(strategy='constant', fill_value=0)),
    #                     ('print', PrintTransformer('imputation'))
    #                 ])),
    #                 ('carryover', Pipeline([
    #                     ('carryover', carryover),
    #                     ('print', PrintTransformer('carryover'))
    #                 ])),
    #                 ('extract_key', Pipeline([
    #                     ('extract_key', ExtractKeyTransformer(carryover_key)),  # Specify the key here
    #                     ('print', PrintTransformer('key extraction'))
    #                 ])),
    #                 ('saturation', Pipeline([
    #                     ('saturation', saturation),
    #                     ('print', PrintTransformer('saturation'))
    #                 ]))
    #             ])
    #         else:
    #             pipeline = Pipeline([
    #                 ('scaler', Pipeline([
    #                     ('minmax', MinMaxScaler()),
    #                     ('print', PrintTransformer('scaling'))
    #                 ])),
    #                 ('imputer', Pipeline([
    #                     ('simple_imputer', SimpleImputer(strategy='constant', fill_value=0)),
    #                     ('print', PrintTransformer('imputation'))
    #                 ]))
    #             ])
    #         transformers.append((col, pipeline, [col]))
    #
    #     adstock= ColumnTransformer(transformers)
    #     return adstock



def create_carryover_pipeline(carryover_model, carryover_params, carryover_key):
    carryover = TransformationPipeline_1().get_carryover_instance(carryover_model, carryover_params)
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Scale the data
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Handle missing values
        ('carryover', carryover),  # Apply the carryover effect
        ('extract_key', ExtractKeyTransformer(carryover_key))  # Extract the necessary key if needed
    ])
    return pipeline