from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import read_data
import numpy as np

file = "players_22.csv"
df = read_data(file)

# Konwersja pozycji na wartości numeryczne
df['main_position'] = df['player_positions'].str.split(',').str[0]
label_encoder = LabelEncoder()
df['main_position_encoded'] = label_encoder.fit_transform(df['main_position'])

selected_cols = ['age'] + ['main_position_encoded'] + [col for col in df.columns if any(
    key in col for key in ['attacking_', 'skill_', 'movement_', 'power_', 'mentality_', 'defending_', 'goalkeeping_'])]
selected_cols.remove('goalkeeping_speed')
selected_cols.remove('skill_moves')

df_model = df[selected_cols + ['overall']].copy()

X = df_model[selected_cols].to_numpy()
y = df_model['overall'].to_numpy().reshape(-1,1)


def closed_form_solution(X, y):
    # theta = (X^T X)^(-1) X^T y
    XtX = X.T @ X
    Xty = X.T @ y
    theta = np.linalg.inv(XtX) @ Xty
    return theta

def gradient_descent(X, y, lr=0.01, epochs=1000, batch_size=32):
    m, n = X.shape
    theta = np.zeros((n, 1))
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = X_batch @ theta
            error = y_pred - y_batch
            gradient = (2 / X_batch.shape[0]) * (X_batch.T @ error)
            theta -= lr * gradient

        if epoch % 100 == 0:
            y_full_pred = X @ theta
            mse = np.mean((y_full_pred - y) ** 2)
            print(f"Epoch {epoch}: MSE = {mse:.2f}")

    return theta

def closed_form_solution_test():
    # Trening własnej implementacji
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

    theta = closed_form_solution(X_train, y_train)
    y_pred_custom = X_test @ theta
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)

    # Trening modelu z scikit-learn
    X_train_sklearn, X_test_sklearn, y_train_sklearn, y_test_sklearn = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train_sklearn, y_train_sklearn)
    y_pred_sklearn = model.predict(X_test_sklearn)
    mse_sklearn = mean_squared_error(y_test_sklearn, y_pred_sklearn)
    r2_sklearn = r2_score(y_test_sklearn, y_pred_sklearn)

    # Wyniki
    print(f"MSE (closed-form): {mse_custom:.2f}")
    print(f"MSE (scikit-learn): {mse_sklearn:.2f}")
    print(f"R2 (closed form): {r2_custom:.2f}")
    print(f"R2 (scikit-learn): {r2_sklearn:.2f}")

    # Współczynniki własnej implementacji
    coef_names = ['bias'] + selected_cols
    print("\nWspółczynniki (closed-form):")
    for name, val in zip(coef_names, theta.flatten()):
        print(f"{name}: {val:.3f}")


def gradient_descent_test():
    # Normalizacja cech
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # Dodaj kolumnę biasu
    X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])

    # Podział na trening/test
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

    # Trening
    theta_gd = gradient_descent(X_train, y_train, lr=0.001, epochs=500, batch_size=64)

    # Predykcja
    y_pred_gd = X_test @ theta_gd
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    r2_gd = r2_score(y_test, y_pred_gd)

    # Wypisanie współczynników
    coef_names = ['bias'] + selected_cols
    print("\nWspółczynniki (gradient descent):")
    for name, val in zip(coef_names, theta_gd.flatten()):
        print(f"{name}: {val:.3f}")

    X_train_sklearn, X_test_sklearn, _, _ = train_test_split(X_norm, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train_sklearn, y_train)
    y_pred_sklearn = model.predict(X_test_sklearn)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)

    print(f"\nFinal MSE (Gradient Descent): {mse_gd:.2f}")
    print(f"Final R2 (Gradient Descent): {r2_gd:.3f}")
    print(f"\nMSE (scikit-learn LinearRegression): {mse_sklearn:.2f}")
    print(f"R2 (scikit-learn LinearRegression): {r2_sklearn:.3f}")


closed_form_solution_test()
gradient_descent_test()