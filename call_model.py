import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
import numpy as np
from data_prep import calculate_val, calculate_vif, DataPrep, data_filter, is_holiday, coefficient
from model_ml import TransformationPipeline_1, create_carryover_pipeline
from hyperparameter_tuning import HyperparameterTuning
from sklearn.model_selection import train_test_split
from parameters_extr import get_carryover_function_parameter_names, get_saturation_model_parameter_names
from prophet_used import ProphetForecasting
from visualisation import visual_plot, share_effect, decay_effect, area_chart

data=pd.read_csv('real_data.csv')
# import pdb;pdb.set_trace()
data.fillna(0,inplace=True)
data_filter(data)
data=data.rename(columns={"DATE_DAY": "ds", "REVENUE": "y"})
data['ds'] = pd.to_datetime(data['ds'])
data['holiday'] = data['ds'].apply(is_holiday)
prophet_forecaster = ProphetForecasting(periods=30)
forecast_df = prophet_forecaster.fit_transform(data[['ds','y']])
data = data.merge(forecast_df,on='ds')
if ('y_y' in data.columns):
    data.pop('y_y')
if ('y_x'in data.columns):
    data = data.rename(columns={'y_x': 'y'})

prep_obj = DataPrep()
data = prep_obj.freq_tranformation(data,"W")
orignal_data=data['y']
x_values=data.index
data=data.iloc[:,1:]
data_2=data.copy()
data.fillna(0,inplace=True)
train_data, test_data = train_test_split(data,train_size=0.8)
p_obj = TransformationPipeline_1()
x_columns=[i for i in data.columns if i!='y']
tuning_obj = HyperparameterTuning(p_obj,train_data, test_data, 'y', x_columns)       # creted object for the hyper parameter tuning class, where we randomly select the combinations of the models and their features.
best_params, best_score = tuning_obj.optimum_parameter_selection()
function_name = best_params['carryover_model']
parameter_names1 = get_carryover_function_parameter_names(function_name)
best_carry_params={}
for i in parameter_names1:
    if i in best_params.keys():
        best_carry_params[i]=best_params[i]
model_name =best_params['saturation_model']
parameter_names = get_saturation_model_parameter_names(model_name)
best_saturation_params={}
for i in parameter_names:
    if i in best_params.keys():
        best_saturation_params[i]=best_params[i]
print(best_params)
x_columns=[]
for i in data.columns:
    if i !='y':
        x_columns.append(i)
adstock = p_obj.column_transformer(x_columns,best_params['carryover_model'],best_params['saturation_model'],best_carry_params,best_saturation_params,carryover_key='x_decayed')
X = adstock.fit_transform(data[x_columns])
data_new = pd.DataFrame(X, columns=x_columns)
target = data['y']
ridge_model = Ridge()
ridge_model.fit(data_new, target)
X=data_new

adstock_data=pd.DataFrame(data_new,columns=X.columns,index=X.index)
weights=pd.Series(ridge_model.coef_,index=X.columns)
base=ridge_model.intercept_
unadj_contributions = adstock_data.mul(weights).assign(Base=base)
adj_contributions = (unadj_contributions
                     .div(unadj_contributions.sum(axis=1), axis=0)
                     .mul(np.array(orignal_data), axis=0))
adj_contributions = adj_contributions.applymap(lambda x: 0.00 if x == -0.00 else x)
print(adj_contributions)
import pdb;pdb.set_trace()


predicted_values = ridge_model.predict(data_new)
calculate_val(data_new,target)
# print(calculate_vif(data_new,x_columns))
print(coefficient(data_new,target))
visual_plot(x_values,orignal_data,predicted_values)
share_effect(data_new,target,orignal_data)
carryover_model = best_params['carryover_model']
carryover_params = best_carry_params
carryover_key = 'sliding_window_cum'
carryover_pipeline = create_carryover_pipeline(carryover_model, carryover_params, carryover_key)
carryover_transformed_data = carryover_pipeline.fit_transform(data_2)
decayed=carryover_transformed_data
# print('value of decay',decayed)
decay_effect(decayed)
import pdb;pdb.set_trace()
ax = (adj_contributions[['GOOGLE_SPEND', 'META_FACEBOOK_SPEND', 'META_INSTAGRAM_SPEND',
       'META_OTHER_SPEND', 'TIKTOK_SPEND', 'Base']].plot.area(
    figsize=(16, 10),
    linewidth=1,
    title='Predicted Sales and Breakdown',
    ylabel='Sales',
    xlabel='Date'
)
)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[::-1], labels[::-1],
    title='Channels', loc="center left",
    bbox_to_anchor=(1.01, 0.5)
)



