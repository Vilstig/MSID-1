import os

import pandas as pd

from basic_statistics import numerical_statistics, categorical_statistics
import charts


def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def age_filter(df):
    df_filtered = df[df['age'] <= 45]
    return df_filtered

def setup_directories():
    main_dir = "charts"
    sub_dirs = ["boxplots", "error_bars", "heatmaps", "histograms", "regression", "violinplots"]

    os.makedirs(main_dir, exist_ok=True)

    for sub in sub_dirs:
        os.makedirs(os.path.join(main_dir, sub), exist_ok=True)

    print("Directories created successfully!")

# noinspection PyPackageRequirements
def main():
    file = "players_22.csv"
    data = read_data(file)

    setup_directories()

    numerical_statistics(data)

    categorical_statistics(data)

    data = age_filter(data)

    boxplot_columns = ["age", "weight_kg"]
    charts.create_boxplots(data, boxplot_columns)

    charts.top_5_leagues_boxplot(data)

    violinplot_columns = ["wage_eur", "overall"]

    charts.create_violinplots(data, violinplot_columns)

    charts.create_error_bars(data)

    histogram_columns = ["overall", "weight_kg", "pace", "shooting", "dribbling", "defending", "physic"]

    charts.create_histograms(data, histogram_columns)

    charts.create_conditioned_histograms(data, histogram_columns[2:])

    charts.create_heatmap_attributes_correlation(data)

    charts.create_heatmap_attributes_positions(data)

    charts.create_heatmap_age_attributes(data)

    charts.create_linear_regression_plot_value(data, "overall")

    charts.create_linear_regression_plot(data, "physic", "defending")

    charts.create_linear_regression_plot(data, "shooting", "defending")

    charts.create_linear_regression_plot(data, "dribbling", "passing")

    charts.create_linear_regression_plot(data, "power_stamina", "movement_sprint_speed")

    charts.create_linear_regression_plot_overall_threshold(data, "age", "power_stamina", 75)

    charts.create_linear_regression_plot_overall_range(data, "age", "movement_acceleration", 75, 8)

    charts.create_linear_regression_plot_release_clause(data, "age", 78)

    charts.create_regression_plot_potential_vs_overall(data)


if __name__ == "__main__":
    main()
