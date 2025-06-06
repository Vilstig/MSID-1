import pandas as pd


def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

# ========== Przewidywanie overallu zawodnika ==========
def predict_player_overall(sofifa_id, model, data, selected_cols):
    sofifa_id = int(sofifa_id)
    player_data = data[data['sofifa_id'] == sofifa_id]

    if player_data.empty:
        return f"Nie znaleziono zawodnika o id {sofifa_id}"

    # Przygotowanie danych wejściowych
    player_input = pd.DataFrame([{
        col: (player_data[col].values[0] if col != 'main_position' else
              player_data['player_positions'].str.split(',').str[0].values[0])
        for col in selected_cols
    }])

    predicted_overall = model.predict(player_input)[0]
    actual_overall = player_data['overall'].values[0]
    player_name = player_data['short_name'].values[0]

    return f"""
Zawodnik: {player_name}
Przewidywany overall: {predicted_overall:.2f}
Rzeczywisty overall: {actual_overall}
"""
