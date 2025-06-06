import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from util import read_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Używane urządzenie:", device)

# Wczytaj dane
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

# Normalizacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na trening i test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Konwersja do tensora
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Definicja modelu klasycznego
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

#Definicja lepszego modelu
class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Model NN, loss i optymalizator
model = NeuralNetworkModel(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Model LR, loss i optymalizator
'''model = LinearRegressionModel(input_dim=X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)'''

# Trening
start = time.time()
epochs = 3000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

end = time.time()

print(f"Czas wykonania: {end - start:.6f} sekundy")

# Ewaluacja
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy()
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nFinal MSE (PyTorch): {mse:.2f}")
    print(f"\nFinal R2 (PyTorch): {r2:.3f}")

