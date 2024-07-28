import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
from data_prep import DataPrep
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin

# ProphetForecasting class definition
class ProphetForecasting(BaseEstimator, TransformerMixin):
    def __init__(self, periods=0, add_seasonality=True, add_quarterly_seasonality=True):
        self.periods = periods
        self.add_seasonality = add_seasonality
        self.add_quarterly_seasonality = add_quarterly_seasonality
        self.model = Prophet()
        if self.add_seasonality:
            self.model.add_seasonality(name='daily', period=1, fourier_order=3)
        if self.add_quarterly_seasonality:
            self.model.add_seasonality(name='quarterly', period=90.25, fourier_order=10)
        self.forecast = None

    def fit(self, X, y=None):
        # Check the columns before fitting
        if 'ds' not in X.columns or 'y' not in X.columns:
            raise ValueError('Dataframe must have columns "ds" and "y" with the dates and values respectively.')
        self.model.fit(X)
        return self

    def transform(self, X):
        future_dates = self.model.make_future_dataframe(periods=self.periods)
        self.forecast = self.model.predict(future_dates)

        seasonality_columns = ['yearly', 'weekly', 'daily','quarterly']
        existing_columns = [col for col in seasonality_columns if col in self.forecast.columns]
        if existing_columns:
            self.forecast['seasonality'] = self.forecast[existing_columns].sum(axis=1)
        else:
            self.forecast['seasonality'] = 0

        result_df = self.forecast[['ds', 'yhat', 'seasonality', 'trend']]
        result_df.columns = ['ds', 'y', 'seasonality', 'trend']
        return result_df



