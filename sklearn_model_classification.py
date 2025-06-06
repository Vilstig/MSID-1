from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import TomekLinks
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC

from util import read_data

# ========== Dane wejściowe ==========
file = "players_22.csv"
df = read_data(file)

def categorize_overall(overall):
    if overall < 58:
        return "very low"
    elif overall < 66:
        return "low"
    elif overall < 74:
        return "medium"
    elif overall < 85:
        return "high"
    else:
        return "very high"

#df['target'] = df['overall'].apply(categorize_overall)
df['target'] = df['work_rate']
df = df.dropna(subset=['target'])

# ========== Wybór kolumn ==========
selected_cols = ['age'] + [
    col for col in df.columns if any(key in col for key in [
        'attacking_', 'skill_', 'movement_', 'power_', 'mentality_', 'defending_', 'goalkeeping_'])
]
for col_to_drop in ['goalkeeping_speed', 'skill_moves']:
    if col_to_drop in selected_cols:
        selected_cols.remove(col_to_drop)

df_model = df[selected_cols + ['target']].copy()
X = df_model.drop(columns='target')
y = df_model['target']

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
def get_classifiers():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVC': SVC()
    }

models = get_classifiers()


# ========== Test z opcją balansowania ==========
def classification_test(resampling_strategy=None):
    print(f"\n--- Test ({resampling_strategy or 'no resampling'}) ---")
    print(f"{'Model':<20} | {'Prec':>6} | {'Recall':>6} | {'F1':>6}")
    print("-" * 55)


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    for name, model in models.items():
        steps = [('preprocessor', preprocessor)]

        if resampling_strategy == "smote":
            steps.append(('sampler', SMOTE(random_state=42)))
        elif resampling_strategy == "tomek":
            steps.append(('sampler', TomekLinks(sampling_strategy='majority')))

        steps.append(('classifier', model))

        pipeline = ImbPipeline(steps)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        print(f"{name:<20} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f}")

        # Dodatkowy szczegółowy raport
        #print("\nSzczegółowy classification_report:")
        #print(classification_report(y_test, y_pred, zero_division=0))

# ========== Przykładowe użycie ==========
print("\n=== Bez balansowania ===")
classification_test()

print("\n=== Z SMOTE (Oversampling) ===")
classification_test(resampling_strategy="smote")

print("\n=== Z TomekLinks (Undersampling) ===")
classification_test(resampling_strategy="tomek")
