import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from util import read_data, predict_player_overall

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
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


preprocessor = build_preprocessor()


# ========== Modele ==========
def get_models():
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'SVR': SVR()
    }


models = get_models()
best_model = None

# ========== Podział na zbiory ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ========== Standardowy test ==========
def standard_test():
    global best_model
    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append((name, r2, mae, mse, rmse))

        if name == "SVR":
            best_model = pipeline

    print(f"{'Model':<20} | {'R^2':>6} | {'MAE':>6} | {'MSE':>10} | {'RMSE':>10}")
    print("-" * 60)
    for name, r2, mae, mse, rmse in results:
        print(f"{name:<20} | {r2:6.3f} | {mae:6.2f} | {mse:10.2f} | {rmse:10.2f}")


# ========== Cross-validation ==========
def cross_validation_test():
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        print(f"\nModel: {name}")
        print("R^2 foldy:", scores)
        print("Średni R^2:", scores.mean())


def polynomial_features_test():
    for degree in range(1, 3):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=degree)),
            ('regressor', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        train_r2 = r2_score(y_train, pipeline.predict(X_train))
        test_r2 = r2_score(y_test, pipeline.predict(X_test))
        print(f"Degree {degree} -> Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")


def compare_sklearn_regularized_models(alpha=1):
    models_regularized = {
        'Linear Regression': LinearRegression(),
        'Ridge (L2)': Ridge(alpha=alpha),
        'Lasso (L1)': Lasso(alpha=alpha, max_iter=10000)
    }

    results = []

    for name, model in models_regularized.items():
        pipeline_regularized = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline_regularized.fit(X_train, y_train)
        y_pred = pipeline_regularized.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        results.append((name, r2, mae, mse, rmse))

    print(f"{'Model':<20} | {'R^2':>6} | {'MAE':>6} | {'MSE':>10} | {'RMSE':>10}")
    print("-" * 60)
    for name, r2, mae, mse, rmse in results:
        print(f"{name:<20} | {r2:6.3f} | {mae:6.2f} | {mse:10.2f} | {rmse:10.2f}")


def ensemble_test():
    # Przygotowanie bazowych modeli
    models = get_models()

    # Voting Regressor
    voting = VotingRegressor(estimators=[
        ('lr', models['Linear Regression']),
        ('tree', models['Decision Tree']),
    ])

    # Stacking Regressor (z Linear Regression jako meta-model)
    stacking = StackingRegressor(
        estimators=[
            ('tree', models['Decision Tree']),
            ('lr', models['Linear Regression'])
        ],
        final_estimator=Ridge()
    )

    ensembles = {
        'Voting Regressor': voting,
        'Stacking Regressor': stacking
    }

    print(f"{'Model':<20} | {'R^2':>6} | {'MAE':>6} | {'MSE':>10} | {'RMSE':>10}")
    print("-" * 60)

    for name, ensemble_model in ensembles.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ensemble_model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"{name:<20} | {r2:6.3f} | {mae:6.2f} | {mse:10.2f} | {rmse:10.2f}")

def mixture_of_experts():
    print("=== Mixture of Experts ===")

    expert_models = {
        'Linear': LinearRegression(),
        'Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }

    # Przetwarzanie danych + eksperci
    expert_pipelines = {
        name: Pipeline([
            ('preprocessor', build_preprocessor()),
            ('regressor', model)
        ])
        for name, model in expert_models.items()
    }

    # Trenuj ekspertów
    print("Trenowanie ekspertów...")
    expert_train_preds = []
    expert_test_preds = []
    
    for name, pipe in expert_pipelines.items():
        pipe.fit(X_train, y_train)
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        
        expert_train_preds.append(train_pred)
        expert_test_preds.append(test_pred)
        
        # Oceń pojedynczego eksperta
        r2 = r2_score(y_test, test_pred)
        mse = mean_squared_error(y_test, test_pred)
        print(f"  {name:>10}: R² = {r2:.3f}  MSE = {mse:.2f}")

    # Konwertuj do numpy arrays
    expert_train_preds = np.column_stack(expert_train_preds)
    expert_test_preds = np.column_stack(expert_test_preds)

    # Przygotuj features dla gating network
    # Używamy zarówno oryginalnych cech jak i predykcji ekspertów
    X_train_processed = expert_pipelines['Linear'].named_steps['preprocessor'].fit_transform(X_train)
    X_test_processed = expert_pipelines['Linear'].named_steps['preprocessor'].transform(X_test)
    
    # Gating network features: oryginalne cechy + predykcje ekspertów
    gating_train_features = np.hstack([
        X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed,
        expert_train_preds
    ])
    gating_test_features = np.hstack([
        X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
        expert_test_preds
    ])

    # Oblicz błędy ekspertów na zbiorze treningowym
    expert_errors = np.abs(expert_train_preds - y_train.values.reshape(-1, 1))
    
    # Odwrotność błędów jako cele dla gating network (mniejszy błąd = większa waga)
    expert_weights = 1.0 / (expert_errors + 1e-8)  # dodajemy małą wartość aby uniknąć dzielenia przez 0

    expert_weights = expert_weights / expert_weights.sum(axis=1, keepdims=True)

    gating_test_weights = []
    
    print("\nTrenowanie gating network...")
    for i in range(len(expert_models)):
        # Trenuj osobny model dla każdego eksperta
        gating_model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        gating_model.fit(gating_train_features, expert_weights[:, i])
        
        # Predykcja wag dla zbioru testowego
        test_weights = gating_model.predict(gating_test_features)
        test_weights = np.clip(test_weights, 0, 1)  # Upewnij się że wagi są nieujemne
        gating_test_weights.append(test_weights)

    # Konwertuj wagi testowe do numpy array i znormalizuj
    gating_test_weights = np.column_stack(gating_test_weights)
    gating_test_weights = gating_test_weights / (gating_test_weights.sum(axis=1, keepdims=True) + 1e-8)

    # Końcowa predykcja: ważona suma ekspertów
    moe_preds = np.sum(expert_test_preds * gating_test_weights, axis=1)

    # Ocena MoE
    r2 = r2_score(y_test, moe_preds)
    mae = mean_absolute_error(y_test, moe_preds)
    mse = mean_squared_error(y_test, moe_preds)
    rmse = np.sqrt(mse)

    print(f"\nWyniki Mixture of Experts:")
    print(f"R²:   {r2:.3f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

def plot_actual_vs_predicted_overall(best_model, X_test, y_test):
    """Tworzy wykres porównujący rzeczywisty i przewidywany overall graczy."""

    # Przewidywanie na podstawie modelu
    y_pred = best_model.predict(X_test)

    # Sortowanie według rzeczywistego overall
    sorted_indices = y_test.argsort()
    y_test_sorted = y_test.iloc[sorted_indices].reset_index(drop=True)
    y_pred_sorted = pd.Series(y_pred).iloc[sorted_indices].reset_index(drop=True)

    # Tworzenie wykresu
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred_sorted, label='Przewidywany overall', linewidth=2, linestyle='--')
    plt.plot(y_test_sorted, label='Rzeczywisty overall', linewidth=2)
    plt.xlabel("Gracze (posortowani po rzeczywistym overallu)")
    plt.ylabel("Overall")
    plt.title("Porównanie rzeczywistego i przewidywanego overallu")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== Wykonanie testów ==========
if __name__ == "__main__":
    standard_test()
    # cross_validation_test()
    # polynomial_features_test()
    # compare_sklearn_regularized_models()
    # Przykładowe ID do testu
    # ensemble_test()
    # mixture_of_experts
    plot_actual_vs_predicted_overall(best_model, X_test, y_test)
    for pid in ["188350", "188545", "230390", "254699", "232505"]:
        print(predict_player_overall(pid, best_model, df, selected_cols))