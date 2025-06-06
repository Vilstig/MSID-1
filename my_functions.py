from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from util import read_data
import numpy as np
import matplotlib.pyplot as plt

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

def compute_mse(X, y, theta):
    y_pred = X @ theta
    return np.mean((y_pred - y) ** 2)

def gradient_descent(X_train, y_train, X_test, y_test, lr=0.01, epochs=1000, batch_size=32, regularization=None, lambda_=0.1):
    m, n = X_train.shape
    theta = np.zeros((n, 1))

    train_costs = []
    test_costs = []

    for epoch in range(epochs):
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            y_pred = X_batch @ theta
            error = y_pred - y_batch
            gradient = (2 / X_batch.shape[0]) * (X_batch.T @ error)

            if regularization == "l2":
                reg_term = 2 * lambda_ * theta
                reg_term[0] = 0  # bias nie jest regularyzowany
                gradient += reg_term
            elif regularization == "l1":
                reg_term = lambda_ * np.sign(theta)
                reg_term[0] = 0
                gradient += reg_term

            theta -= lr * gradient

        # Obliczamy i zapisujemy MSE po każdej epoce
        train_mse = compute_mse(X_train, y_train, theta)
        test_mse = compute_mse(X_test, y_test, theta)
        train_costs.append(train_mse)
        test_costs.append(test_mse)

    return theta, train_costs, test_costs


def plot_cost(train_costs, test_costs):
    import numpy as np
    import matplotlib.pyplot as plt

    epochs = np.arange(len(train_costs))
    train_costs = np.array(train_costs)
    test_costs = np.array(test_costs)

    plt.figure(figsize=(12, 6))

    # Rysowanie funkcji kosztu
    plt.plot(epochs, train_costs, label='Train MSE', color='blue', linewidth=2)
    plt.plot(epochs, test_costs, label='Test MSE', color='orange', linewidth=2, linestyle='--')

    # Skala logarytmiczna na osi Y
    plt.yscale('log')

    plt.xlabel('Epoka')
    plt.ylabel('Średni błąd kwadratowy (MSE, skala log)')
    plt.title('Zbieżność funkcji kosztu podczas treningu (logarytmiczna skala)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


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


def gradient_descent_test(show_plot=False):
    # Normalizacja cech
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # Dodaj kolumnę biasu
    X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])

    # Podział na trening/test
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

    # Trening
    theta_gd, train_costs, test_costs = gradient_descent(X_train, y_train, X_test, y_test, lr=0.0001, epochs=100, batch_size=64)

    #Wyświetl wykres funkcji kosztu
    if show_plot:
        plot_cost(train_costs, test_costs)

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


def cross_validate_closed_form(X_data, y_data, k=3):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []

    fold = 1
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        theta = closed_form_solution(X_train, y_train)
        y_pred = X_test @ theta

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"[Fold {fold}] MSE: {mse:.2f}, R2: {r2:.3f}")
        fold += 1

    print("\nŚrednie wyniki (closed-form):")
    print(f"Średni MSE: {np.mean(mse_scores):.2f}")
    print(f"Średni R²: {np.mean(r2_scores):.3f}")


def cross_validate_gradient_descent(X_data, y_data, k=3, lr=0.001, epochs=500, batch_size=64):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    r2_scores = []

    fold = 1
    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        theta, _, _ = gradient_descent(X_train, y_train, lr=lr, epochs=epochs, batch_size=batch_size)
        y_pred = X_test @ theta

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"[Fold {fold}] MSE: {mse:.2f}, R2: {r2:.3f}")
        fold += 1

    print("\nŚrednie wyniki (gradient descent):")
    print(f"Średni MSE: {np.mean(mse_scores):.2f}")
    print(f"Średni R²: {np.mean(r2_scores):.3f}")

def cross_validate_test():
    # Przygotowanie danych z biasem
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    cross_validate_closed_form(X_bias, y)

    # Przygotowanie danych z normalizacją do GD
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    X_bias_norm = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])
    cross_validate_gradient_descent(X_bias_norm, y)

def regularization_test():
    # Normalizacja cech
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean) / X_std

    # Dodaj kolumnę biasu
    X_bias = np.hstack([np.ones((X_norm.shape[0], 1)), X_norm])

    # Podział na trening/test
    X_train, X_test, y_train, y_test = train_test_split(X_bias, y, test_size=0.2, random_state=42)

    regularizations = [None, "l2", "l1"]
    results = {}

    for reg in regularizations:
        label = "No regularization" if reg is None else reg.upper()
        print(f"\n===== {label} =====")

        theta, _, _ = gradient_descent(
            X_train, y_train, X_test, y_test,
            lr=0.0001, epochs=100, batch_size=64,
            regularization=reg, lambda_=0.1
        )
        y_pred = X_test @ theta
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[label] = theta.flatten()

        # Wypisanie wyników
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.3f}")

    # Porównanie wag
    print("\n=== PORÓWNANIE WAG ===")
    headers = ['Atrybut'] + list(results.keys())
    row_format = "{:<30}" + " {:>15}" * len(results)
    print(row_format.format(*headers))
    print("-" * (30 + 15 * len(results)))

    for i, name in enumerate(['bias'] + selected_cols):
        row = [name] + [f"{results[reg][i]:.3f}" for reg in results]
        print(row_format.format(*row))


#closed_form_solution_test()
#gradient_descent_test(True)

#cross_validate_test()

regularization_test()
