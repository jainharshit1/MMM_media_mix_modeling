import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class HyperparameterTuning:
    def __init__(self, transformation, train_data, test_data, target, explanatory_vars):
        self.transformation = transformation
        self.train_data = train_data
        self.test_data = test_data
        self.target = target
        self.explanatory_vars = explanatory_vars

    def objective_function(self, trial):
        carryover_model = trial.suggest_categorical('carryover_model',['ExponentialCarryover','GaussianCarryover'])

        saturation_model = trial.suggest_categorical('saturation_model',
                                                      ['ExponentialSaturation','BoxCoxSaturation','HillSaturation','AdbudgSaturation'])

        regression_model = trial.suggest_categorical('regression_model',['Ridge'])


        carryover_params = {}
        if carryover_model == 'ExponentialCarryover':
            carryover_params['length'] = trial.suggest_int('length', 0, 6)
            carryover_params['strength'] = trial.suggest_float('strength', 0.0, 1.0)

        elif carryover_model == 'GaussianCarryover':
            carryover_params['window'] = trial.suggest_int('window', 10, 300)
            carryover_params['p'] = trial.suggest_float('p', 0, 1)
            carryover_params['sig'] = trial.suggest_float('sig', 0.0, 1.0)


        saturation_params = {}
        if saturation_model == 'ExponentialSaturation':
            saturation_params['a'] = trial.suggest_float('a', 0.0 ,1.0)

        elif saturation_model == 'BoxCoxSaturation':
            saturation_params['exponent'] = trial.suggest_float('exponent', 0.0,1.0)
            saturation_params['shift'] = trial.suggest_float('shift', 0.0, 1.0)

        elif saturation_model == 'HillSaturation':
            saturation_params['exponent'] = trial.suggest_float('exponent', 0.0, 1.0)
            saturation_params['half_saturation'] = trial.suggest_float('half_saturation', 0.0,1.0)

        elif saturation_model == 'AdbudgSaturation':
            saturation_params['exponent'] = trial.suggest_float('exponent', 0.0,1.0)
            saturation_params['denominator_shift'] = trial.suggest_float('denominator_shift', 0.0,1.0)

        column_transformer = self.transformation.column_transformer(
            self.explanatory_vars, carryover_model, saturation_model, carryover_params, saturation_params,carryover_key='x_decayed')
        adstock_train_data = column_transformer.fit_transform(self.train_data[self.explanatory_vars])
        adstock_test_data = column_transformer.transform(self.test_data[self.explanatory_vars])

        try:
            ridge_model = Ridge()
            ridge_model.fit(adstock_train_data, self.train_data[self.target])
            preds = ridge_model.predict(adstock_test_data)
        except:
            import pdb;pdb.set_trace()
        mse = mean_squared_error(self.test_data[self.target], preds)
        return mse

    def optimum_parameter_selection(self, n_trials=1000):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective_function, n_trials=n_trials)

        best_params = study.best_params
        best_loss = study.best_value
        print("Best Parameters:", best_params)
        print("Best Loss:", best_loss)
        return best_params, best_loss
