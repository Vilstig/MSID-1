import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils import read_data
import pandas as pd

file = "players_22.csv"
df = read_data(file)


df['main_position'] = df['player_positions'].str.split(',').str[0]
selected_cols = ['age'] + ['main_position'] + [col for col in df.columns if any(
    key in col for key in ['attacking_', 'skill_', 'movement_', 'power_', 'mentality_', 'defending_', 'goalkeeping_'])]
selected_cols.remove('goalkeeping_speed')
selected_cols.remove('skill_moves')


df_model = df[selected_cols + ['overall']].copy()


X = df_model.drop(columns=['overall'])
y = df_model['overall']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline dla danych kategorycznych
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Połączone przetwarzanie kolumn
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVR': SVR()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)   #Średni błąd bezwzględny
    mse = mean_squared_error(y_test, y_pred)    #Średni błąd kwadratowy
    rmse = np.sqrt(mse)                         #Pierwiastek z mse
    results.append((name, r2, mae, mse, rmse))

    if name == "SVR":
        best_model = pipeline

# Wyświetl wyniki
title = f"{'Model':<20} | {'R^2':>6} | {'MAE':>6} | {'MSE':>10} | {'RMSE':>10}"
print(title)
print("-" * len(title))
for name, r2, mae, mse, rmse in results:
    print(f"{name:<20} | {r2:6.3f} | {mae:6.2f} | {mse:10.2f} | {rmse:10.2f}")



def predict_player_overall(sofifa_id):
    # Wczytanie danych zawodnika
    sofifa_id = int(sofifa_id)
    player_data = df[df['sofifa_id'] == sofifa_id]
    
    if player_data.empty:
        return f"Nie znaleziono zawodnika o id {sofifa_id}"

    player_name = player_data['short_name']

    # Przygotowanie danych wejściowych dla modelu
    player_input = pd.DataFrame([{
        'age': player_data['age'].values[0],
        'main_position': player_data['player_positions'].str.split(',').str[0].values[0],
        'attacking_crossing': player_data['attacking_crossing'].values[0],
        'attacking_finishing': player_data['attacking_finishing'].values[0],
        'attacking_heading_accuracy': player_data['attacking_heading_accuracy'].values[0],
        'attacking_short_passing': player_data['attacking_short_passing'].values[0],
        'attacking_volleys': player_data['attacking_volleys'].values[0],
        'skill_dribbling': player_data['skill_dribbling'].values[0],
        'skill_curve': player_data['skill_curve'].values[0],
        'skill_fk_accuracy': player_data['skill_fk_accuracy'].values[0],
        'skill_long_passing': player_data['skill_long_passing'].values[0],
        'skill_ball_control': player_data['skill_ball_control'].values[0],
        'movement_acceleration': player_data['movement_acceleration'].values[0],
        'movement_sprint_speed': player_data['movement_sprint_speed'].values[0],
        'movement_agility': player_data['movement_agility'].values[0],
        'movement_reactions': player_data['movement_reactions'].values[0],
        'movement_balance': player_data['movement_balance'].values[0],
        'power_shot_power': player_data['power_shot_power'].values[0],
        'power_jumping': player_data['power_jumping'].values[0],
        'power_stamina': player_data['power_stamina'].values[0],
        'power_strength': player_data['power_strength'].values[0],
        'power_long_shots': player_data['power_long_shots'].values[0],
        'mentality_aggression': player_data['mentality_aggression'].values[0],
        'mentality_interceptions': player_data['mentality_interceptions'].values[0],
        'mentality_positioning': player_data['mentality_positioning'].values[0],
        'mentality_vision': player_data['mentality_vision'].values[0],
        'mentality_penalties': player_data['mentality_penalties'].values[0],
        'mentality_composure': player_data['mentality_composure'].values[0],
        'defending_marking_awareness': player_data['defending_marking_awareness'].values[0],
        'defending_standing_tackle': player_data['defending_standing_tackle'].values[0],
        'defending_sliding_tackle': player_data['defending_sliding_tackle'].values[0],
        'goalkeeping_diving': player_data['goalkeeping_diving'].values[0],
        'goalkeeping_handling': player_data['goalkeeping_handling'].values[0],
        'goalkeeping_kicking': player_data['goalkeeping_kicking'].values[0],
        'goalkeeping_positioning': player_data['goalkeeping_positioning'].values[0],
        'goalkeeping_reflexes': player_data['goalkeeping_reflexes'].values[0]
    }])
    
    # Przewidywanie overall'u
    predicted_overall = best_model.predict(player_input)[0]
    actual_overall = player_data['overall'].values[0]
    
    return f"""
Zawodnik: {player_name}
Przewidywany overall: {predicted_overall:.2f}
Rzeczywisty overall: {actual_overall}
"""

'''
# Przykład użycia:
print(predict_player_overall("188350"))
print(predict_player_overall("188545"))
print(predict_player_overall("230390"))
print(predict_player_overall("254699"))
print(predict_player_overall("232505"))
'''