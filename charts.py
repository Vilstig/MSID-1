import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.version import release

top_5_leagues = ["English Premier League", "Spain Primera Division", "Italian Serie A", "German 1. Bundesliga",
                 "French Ligue 1"]

position_mapping = {
    "GK": "Bramkarz", "CB": "Obrońca", "LCB": "Obrońca", "RCB": "Obrońca", "LB": "Obrońca", "RB": "Obrońca",
    "LWB": "Obrońca", "RWB": "Obrońca", "CDM": "Pomocnik", "CM": "Pomocnik", "LCM": "Pomocnik", "RCM": "Pomocnik",
    "LM": "Pomocnik", "RM": "Pomocnik", "CAM": "Pomocnik", "LAM": "Pomocnik", "RAM": "Pomocnik", "LDM": "Pomocnik",
    "RDM": "Pomocnik", "LW": "Napastnik", "RW": "Napastnik", "ST": "Napastnik", "LS": "Napastnik",
    "RS": "Napastnik",
    "CF": "Napastnik", "LF": "Napastnik", "RF": "Napastnik"
}

player_attributes = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]

player_full_attributes = ["overall", "potential", "value_eur", "wage_eur", "age", "height_cm", "weight_kg", "pace", "shooting", "passing", "dribbling", "defending", "physic"]

def create_boxplots(df, columns):
    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[col])
            plt.title(f"Boxplot dla {col}")
            plt.ylabel(col)
            plt.savefig(f"charts/boxplots/boxplot_{col}.png")
            plt.close()
            print(f"Zapisano boxplot dla {col} jako boxplot_{col}.png")


def top_5_leagues_boxplot(df):
    df_top_leagues = df[df["league_name"].isin(top_5_leagues)]

    plt.figure(figsize=(18, 9))
    sns.boxplot(x=df_top_leagues["league_name"], y=df_top_leagues["overall"])
    plt.title("Boxplot ocen ogólnych piłkarzy w top 5 ligach")
    plt.xlabel("Liga")
    plt.ylabel("Overall")
    plt.savefig("charts/boxplots/boxplot_top5_leagues.png")
    plt.close()
    print("Zapisano boxplot dla piłkarzy z top 5 lig jako boxplot_top5_leagues.png")


def create_violinplots(df, columns):
    df_top_leagues = df[df["league_name"].isin(top_5_leagues)]

    df_copy = df.copy()
    df_top_leagues_copy = df_top_leagues.copy()

    df_copy["Grupa"] = "Wszyscy"
    df_top_leagues_copy["Grupa"] = "Top 5 Lig"

    combined_df = pd.concat([df_copy, df_top_leagues_copy], ignore_index=True)

    for col in columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=col, hue="Grupa", data=combined_df, split=True)
        plt.title(f"Violinplot dla {col} (wszyscy vs top 5 lig)")
        plt.ylabel(col)
        plt.savefig(f"charts/violinplots/violinplot_{col}_comparison.png")
        plt.close()


def create_error_bars(df):
    plt.figure(figsize=(8, 6))
    sns.pointplot(x="skill_moves", y="dribbling", data=df, errorbar="sd", capsize=0.1)
    plt.title("Średnia dribbling względem skill moves")
    plt.savefig("charts/error_bars/errorbars_dribbling_skill_moves.png")
    plt.close()

    position_mapping_attackers = {
        "LW": "Attacker", "RW": "Attacker", "ST": "Attacker",
        "LS": "Attacker", "RS": "Attacker", "CF": "Attacker",
        "LF": "Attacker", "RF": "Attacker"
    }

    df["Position_Group"] = df["club_position"].map(position_mapping_attackers)
    df_filtered = df.dropna(subset=["Position_Group"])

    df_filtered = df_filtered[(df_filtered["overall"] >=  70) & (df_filtered["overall"] <= 80)]

    df_melted = df_filtered.melt(id_vars=["Position_Group"], value_vars=player_attributes,
                                 var_name="Attribute", value_name="Value")

    plt.figure(figsize=(12, 8))
    sns.pointplot(x="Attribute", y="Value", data=df_melted, errorbar="sd", capsize=0.1)

    plt.title("Error Bars for Attacker Attributes (70-80 overall)")
    plt.xticks(rotation=45)
    plt.savefig("charts/error_bars/errorbars_attackers_attributes.png")
    plt.close()



def create_histograms(df, columns):
    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col].dropna(), bins=20, kde=True)
        plt.title(f"Histogram dla {col}")
        plt.savefig(f"charts/histograms/histogram_{col}.png")
        plt.close()


def create_conditioned_histograms(df, columns):
    position_mapping_copy = position_mapping.copy()
    position_mapping_copy.pop("GK", None)

    df["Position_Group"] = df["club_position"].map(position_mapping_copy)
    df_filtered = df.dropna(subset=["Position_Group"])

    for col in columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df_filtered, x=col, bins=20, kde=True, hue="Position_Group", element="step")
        plt.title(f"Histogram dla {col} według pozycji")
        plt.savefig(f"charts/histograms/histogram_{col}_by_position.png")
        plt.close()


def create_heatmap_attributes_correlation(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[player_full_attributes].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.xticks(rotation=45, fontweight="bold", ha="right")
    plt.yticks(fontweight="bold")
    plt.title("Macierz korelacji między atrybutami")
    plt.savefig("charts/heatmaps/heatmap_correlations.png")
    plt.close()
    print("Zapisano heatmap dla korelacji między atrybutami.")


def create_heatmap_attributes_positions(df):
    df_filtered = df[df["overall"] >= 75].copy()

    position_mapping_copy = position_mapping.copy()
    position_mapping_copy.pop("GK", None)

    df_filtered.loc[:, "Position_Group"] = df_filtered["club_position"].map(position_mapping_copy)

    df_filtered = df_filtered.dropna(subset=["Position_Group"])

    df_position_avg = df_filtered.groupby("Position_Group")[player_attributes].mean()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_position_avg, annot=True, cmap="crest", fmt=".1f")
    plt.title("Średnia wartości atrybutów dla różnych pozycji (ocena 75+)")
    plt.ylabel("Pozycja")
    plt.xlabel("Atrybuty")
    plt.savefig("charts/heatmaps/heatmap_attributes_by_position_75+.png")
    plt.close()
    print("Zapisano heatmap dla atrybutów według pozycji (ocena 75+).")

def create_heatmap_age_attributes(df):
    bins = [16, 24, 29, 34, 39, 44]
    df['Age_Group'] = pd.cut(df['age'], bins=bins, labels=["16-23", "25-29", "30-34", "35-39", "40-44"])

    df_age_avg = df.groupby("Age_Group")[player_attributes].mean()

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_age_avg, annot=True, fmt=".1f")
    plt.title("Średnia wartości atrybutów według grup wiekowych")
    plt.ylabel("Grupa wiekowa")
    plt.xlabel("Atrybuty")
    plt.savefig("charts/heatmaps/heatmap_attributes_by_age.png")
    plt.close()
    print("Zapisano heatmap dla atrybutów według grup wiekowych.")


def prepare_data_for_regression(df, x_col, y_col, filter_column=None, threshold=None):
    df_filtered = df.copy()

    if filter_column and threshold:
        df_filtered = df_filtered[df_filtered[filter_column] >= threshold]

    df_filtered = df_filtered.dropna(subset=[y_col])

    if y_col == "release_clause_eur":
        df_filtered["release_clause_log"] = df_filtered[y_col].apply(lambda x: np.log1p(x))
        y_col = "release_clause_log"

    if y_col == "value_eur":
        df_filtered["value_eur"] = df_filtered[y_col].apply(lambda x: np.log1p(x))
        y_col = "value_eur"

    if x_col == "age":
        df_filtered = df_filtered[df_filtered[x_col] <= 45]

    return df_filtered, y_col


def create_linear_regression_plot(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df[x_col], y=df[y_col], scatter_kws={'alpha': 0.5, 's': 20}, line_kws={"color": "red"})

    plt.xlabel(x_col, fontsize=12, fontweight="bold")
    plt.ylabel(y_col, fontsize=12, fontweight="bold")
    plt.title(f"Korelacja liniowa: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

    plt.savefig(f"charts/regression/regplot_{x_col}_vs_{y_col}.png")
    plt.close()

    print(f"Zapisano wykres regresji dla {x_col} vs {y_col}.")


def create_linear_regression_plot_release_clause(df, x_col, overall=75):
    df_filtered, y_col = prepare_data_for_regression(df, x_col, "release_clause_eur", filter_column="overall",
                                                     threshold=overall)

    create_linear_regression_plot(df_filtered, x_col, y_col)

def create_linear_regression_plot_value(df, x_col):
    df_filtered, y_col = prepare_data_for_regression(df, x_col, "value_eur")

    create_linear_regression_plot(df_filtered, x_col, y_col)

def create_linear_regression_plot_overall_threshold(df, x_col, y_col, overall):
    df_filtered, y_col = prepare_data_for_regression(df, x_col, y_col, filter_column="overall", threshold=overall)

    create_linear_regression_plot(df_filtered, x_col, y_col)


def create_regression_plot_potential_vs_overall(df):
    # Compute the difference between potential and overall
    df["potential_gap"] = df["potential"] - df["overall"]
    df_filtered = df[df["age"] < 30]

    # Create the regression plot
    plt.figure(figsize=(8, 6))
    sns.regplot(x=df_filtered["age"], y=df_filtered["potential_gap"], scatter_kws={'alpha': 0.5}, line_kws={"color": "red"})

    # Labels and title
    plt.xlabel("Age", fontsize=12, fontweight="bold")
    plt.ylabel("Potential - Overall", fontsize=12, fontweight="bold")
    plt.title("Regression: Potential vs Age", fontsize=14, fontweight="bold")

    # Save the plot
    plt.savefig("charts/regression/regplot_potential_gap_vs_age.png")
    plt.close()

    print("Zapisano wykres regresji dla różnicy potencjału i overalla względem wieku.")


def create_linear_regression_plot_overall_range(df, x_col, y_col, overall, over_range):
    df_filtered = df[(df["overall"] >= overall - over_range) & (df["overall"] <= overall + over_range)]


    create_linear_regression_plot(df_filtered, x_col, y_col)