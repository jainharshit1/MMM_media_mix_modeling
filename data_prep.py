import pandas as pd
import itertools
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
import pandas as pd


holiday_dates = ["01/01","19/06","04/07","11/11","25/12","31/03","12/02","09/09"]
def is_holiday(date):
    date_str = date.strftime('%d/%m')
    return 1 if date_str in holiday_dates else 0
def data_filter(data):
    columns_to_drop = [
        'GOOGLE_PAID_SEARCH_SPEND', 'GOOGLE_SHOPPING_SPEND', 'GOOGLE_PMAX_SPEND',
        'GOOGLE_DISPLAY_SPEND', 'GOOGLE_VIDEO_SPEND','META_SPEND',
        'GOOGLE_PAID_SEARCH_CLICKS', 'GOOGLE_SHOPPING_CLICKS', 'GOOGLE_PMAX_CLICKS',
        'GOOGLE_DISPLAY_CLICKS', 'GOOGLE_VIDEO_CLICKS', 'META_FACEBOOK_CLICKS',
        'META_INSTAGRAM_CLICKS', 'META_OTHER_CLICKS', 'GOOGLE_PAID_SEARCH_IMPRESSIONS',
        'GOOGLE_SHOPPING_IMPRESSIONS', 'GOOGLE_PMAX_IMPRESSIONS', 'GOOGLE_VIDEO_IMPRESSIONS',
        'GOOGLE_DISPLAY_IMPRESSIONS', 'META_FACEBOOK_IMPRESSIONS', 'META_INSTAGRAM_IMPRESSIONS',
        'META_OTHER_IMPRESSIONS', 'TIKTOK_IMPRESSIONS','TIKTOK_CLICKS'
    ]
    data.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    items_to_remove = ['FIRST_PURCHASES', 'FIRST_PURCHASES_UNITS', 'FIRST_PURCHASES_ORIGINAL_PRICE',
                       'FIRST_PURCHASES_GROSS_DISCOUNT', 'ALL_PURCHASES', 'ALL_PURCHASES_UNITS',
                       'ALL_PURCHASES_ORIGINAL_PRICE', 'ALL_PURCHASES_GROSS_DISCOUNT']
    data.drop(columns=items_to_remove, errors='ignore', inplace=True)


class DataPrep:
    def interaction(self,data,explanatory_vars):
        for var in list(itertools.combinations(explanatory_vars,2)):
            data[f'{var[0]}_{var[1]}'] = data[f'{var[0]}'] * data[f'{var[1]}']
            explanatory_vars.append(f'{var[0]}_{var[1]}')
        return data, explanatory_vars
    def freq_tranformation(self,data,freq):
        if freq == "W":
            data = data.resample('W-Mon', label='left', closed='left', on='ds').sum()
        return data


def calculate_vif(data,explanatory_vars):
    data.fillna(0)
    data_filtered = data[explanatory_vars]
    corr = data_filtered.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()
    data_with_const = add_constant(data_filtered)
    # Calculate VIF for each variable
    vif_data = pd.DataFrame()
    vif_data['Variable'] = data_with_const.columns
    vif_data['VIF'] = [variance_inflation_factor(data_with_const.values, i) for i in range(data_with_const.shape[1])]

    return vif_data

def calculate_val(data_new,target):
    ridge_model = Ridge(positive=True)
    ridge_model.fit(data_new, target)
    kf = KFold(n_splits=5)
    ts = TimeSeriesSplit(5)
    # Perform cross-validation
    scores = []
    for train_index, test_index in kf.split(data_new):
        X_train, X_test = data_new.iloc[train_index], data_new.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]
        ridge_model.fit(X_train, y_train)
        score = ridge_model.score(X_test, y_test)  # Example of using a scoring method
        scores.append(score)
    scores_time = []
    for train_index, test_index in ts.split(data_new):
        X_train, X_test = data_new.iloc[train_index], data_new.iloc[test_index]
        y_train, y_test = target[train_index], target[test_index]
        ridge_model.fit(X_train, y_train)
        score = ridge_model.score(X_test, y_test)  # Example of using a scoring method
        scores_time.append(score)
    print("Cross-validation scores:", scores)
    print("Cross-validation scores:", scores_time)
    ###################
    print('cross-val score: ', cross_val_score(ridge_model, data_new, target, cv=TimeSeriesSplit()))
    print('cross-val score: ', cross_val_score(ridge_model, data_new, target, cv=TimeSeriesSplit()).mean())

def coefficient(data_new,target) :
    ridge_model = Ridge()
    ridge_model.fit(data_new, target)
    coef = ridge_model.coef_
    coef_df=pd.DataFrame({'Feature': data_new.columns, 'Coefficient': coef})
    coef_df.loc[len(coef_df)] = ['intercept', ridge_model.intercept_]
    return(coef_df)