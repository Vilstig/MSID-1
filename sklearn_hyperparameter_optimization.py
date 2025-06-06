import numpy as np
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import DecisionTreeRegressor

from util import read_data

# ========== Dane wejściowe ==========
file = "players_22.csv"
df = read_data(file)
df['main_position'] = df['player_positions'].str.split(',').str[0]

# ========== Wybór kolumn ==========
selected_cols = ['age', 'main_position'] + [
    col for col in df.columns if any(key in col for key in [
        'attacking_', 'skill_', 'movement_', 'power_', 'mentality_', 'defending_', 'goalkeeping_'])
]
for col_to_drop in ['goalkeeping_speed', 'skill_moves']:
    if col_to_drop in selected_cols:
        selected_cols.remove(col_to_drop)

df_model = df[selected_cols + ['overall']].copy()
X = df_model.drop(columns='overall')
y = df_model['overall']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns


# ========== Preprocessing ==========
def build_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


preprocessor = build_preprocessor()


# ========== Modele ==========
def get_regressors():
    return {
        'Ridge Regression': Ridge(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'SVR': SVR()
    }

models = get_regressors()

for name, model in models.items():
    print(model.get_params())

# ========== Test z opcją balansowania ==========
def grid_search_test():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    for name, model in models.items():
        print(f"\nTestowanie modelu: {name}")
        base_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', clone(model))
        ])

        # ==== Trening bazowy (bez GridSearch) ====
        base_pipeline.fit(X_train, y_train)
        base_pred = base_pipeline.predict(X_test)

        base_r2 = r2_score(y_test, base_pred)
        base_mae = mean_absolute_error(y_test, base_pred)
        base_mse = mean_squared_error(y_test, base_pred)
        base_rmse = np.sqrt(base_mse)

        # ==== GridSearch ====
        param_grid = {}
        if name == 'Ridge Regression':
            param_grid = {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
                'regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag']
            }
        elif name == 'Decision Tree':
            param_grid = {
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4],
                'regressor__max_features': [None, 'sqrt', 'log2']
            }
        elif name == 'SVR':
            param_grid = {
                'regressor__kernel': ['linear', 'rbf', 'poly'],
                'regressor__C': [0.1, 1, 10],
                'regressor__gamma': ['scale', 'auto']
            }

        tuned_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', clone(model))
        ])

        grid_search = HalvingGridSearchCV(
            tuned_pipeline,
            param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        tuned_pred = best_model.predict(X_test)

        tuned_r2 = r2_score(y_test, tuned_pred)
        tuned_mae = mean_absolute_error(y_test, tuned_pred)
        tuned_mse = mean_squared_error(y_test, tuned_pred)
        tuned_rmse = np.sqrt(tuned_mse)

        print(f"Najlepsze parametry: {grid_search.best_params_}")

        results.append((name, base_r2, tuned_r2, base_mae, tuned_mae, base_mse, tuned_mse))

    # ===== Wyniki =====
    print(
        f"\n{'Model':<20} | {'R2 (Base)':>10} | {'R2 (Tuned)':>10} | {'MAE ↓':>10} | {'MAE ↓':>10} | {'MSE ↓':>10} | {'MSE ↓':>10}")
    print("-" * 80)
    for name, r2_b, r2_t, mae_b, mae_t, mse_b, mse_t in results:
        print(
            f"{name:<20} | {r2_b:10.3f} | {r2_t:10.3f} | {mae_b:10.2f} | {mae_t:10.2f} | {mse_b:10.2f} | {mse_t:10.2f}")

grid_search_test()