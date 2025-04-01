import pandas as pd

def numerical_statistics(df):
    print("\n### Statystyki opisowe dla cech numerycznych ###")
    numerical_cols = df.select_dtypes(include=['number'])
    stats = []
    for col in numerical_cols:
        stats.append({
            "Kolumna": col,
            "Średnia": numerical_cols[col].mean(),
            "Mediana": numerical_cols[col].median(),
            "Min": numerical_cols[col].min(),
            "Max": numerical_cols[col].max(),
            "Odchylenie standardowe": numerical_cols[col].std(),
            "5-ty percentyl": numerical_cols[col].quantile(0.05),
            "95-ty percentyl": numerical_cols[col].quantile(0.95),
            "Brakujące wartości": numerical_cols[col].isnull().sum()
        })
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv("numerical_statistics.csv", index=False)
    print("Zapisano statystyki numeryczne do pliku numerical_statistics.csv")



def categorical_statistics(df):
    print("\n### Statystyki opisowe dla cech kategorialnych ###")
    categorical_cols = df.select_dtypes(include=['object'])
    stats = []
    for col in categorical_cols:
        stats.append({
            "Kolumna" : col,
            "Unikalne wartości": categorical_cols[col].nunique(),
            "Brakujące wartości": categorical_cols[col].isnull().sum(),
            "Proporcja klas" :categorical_cols[col].value_counts(normalize=True)
        })
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv("categorical_statistics.csv", index=False)
    print("Zapisano statystyki kategorialne do pliku categorical_statistics.csv")

