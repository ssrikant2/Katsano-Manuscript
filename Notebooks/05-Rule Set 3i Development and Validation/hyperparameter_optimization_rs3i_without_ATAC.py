import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.metrics import mean_squared_error

# import data
modelTrainingPath = "../../Data/modelTraining/"
train_df = pd.read_csv(f'{modelTrainingPath}trainData_rs3i.csv')

X = train_df[['sgRNA_\'Cut\'_Site_TSS_Offset', 'rs3ChenSeqScore']].copy()
y = train_df['doubleZscore']
groups = train_df['Target_Gene_Symbol']

def main():
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',  
            'eval_metric': 'rmse',  # Root Mean Squared Error
            'booster': 'gbtree',
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 3, 100)
        }

        X_train = X.copy()  

        # group cv into 4 fold
        gkf = GroupKFold(n_splits=4)  

        rmse_list = []
        for train_index, test_index in gkf.split(X_train, y, groups=groups):
            # QC: divde by group (gene id)
            train_groups = groups[train_index]
            test_groups = groups[test_index]
            unq_train_cv = len(np.unique(train_groups))
            unq_test_cv = len(np.unique(test_groups))
            unq_cv = unq_test_cv + unq_train_cv
            print(f"  Unique groups in CV: {unq_cv}")

            # use that to do cv and fit using the optuna search hyperparameters
            X_train_split, X_test_split = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_split, y_test_split = y.iloc[train_index], y.iloc[test_index]
            model = xgb.XGBRegressor(**param)
            model.fit(X_train_split, y_train_split)

            preds = model.predict(X_test_split)
            rmse = mean_squared_error(y_test_split, preds, squared=False)  # RMSE calculation
            rmse_list.append(rmse)
            print(rmse)

        # mean RMSE for the current set of hyperparameters
        print(len(rmse_list))
        return sum(rmse_list) / len(rmse_list)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=1))  # Minimize RMSE with Tree-structured Parzen Estimator

    study.optimize(objective, n_trials=100, show_progress_bar=True)

    with open(f'{modelTrainingPath}best_hyperparams_rs3i_without_atac.txt', 'w') as f:
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == '__main__':
    main()
