import pandas as pd

from basic_statistics import numerical_statistics, categorical_statistics
from charts import create_boxplots, top_5_leagues_boxplot, create_violinplots, create_error_bars, create_histograms, \
    create_conditioned_histograms, create_heatmap_attributes_correlation, create_heatmap_attributes_positions, \
    create_heatmap_age_attributes, create_linear_regression_plot, create_linear_regression_plot_release_clause, \
    create_linear_regression_plot_overall_threshold


def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)

def age_filter(df):
    df_filtered = df[df['age'] <= 45]
    return df_filtered

def main():
    file = "players_22.csv"
    data = read_data(file)

    numerical_statistics(data)

    categorical_statistics(data)

    data = age_filter(data)

    boxplot_columns = ["age", "weight_kg"]
    create_boxplots(data, boxplot_columns)

    top_5_leagues_boxplot(data)

    violinplot_columns = ["wage_eur", "overall"]

    create_violinplots(data, violinplot_columns)

    create_error_bars(data)

    histogram_columns = ["overall", "weight_kg", "pace", "shooting", "dribbling", "defending", "physic"]

    create_histograms(data, histogram_columns)

    create_conditioned_histograms(data, histogram_columns[2:])

    create_heatmap_attributes_correlation(data)

    create_heatmap_attributes_positions(data)

    create_heatmap_age_attributes(data)

    create_linear_regression_plot(data,"age", "overall")

    create_linear_regression_plot(data,"age", "height_cm")

    create_linear_regression_plot(data,"physic", "defending")

    create_linear_regression_plot(data,"shooting", "defending")

    create_linear_regression_plot(data,"power_stamina", "movement_sprint_speed")

    create_linear_regression_plot_overall_threshold(data,"age", "power_stamina", 75)

    create_linear_regression_plot_overall_threshold(data,"physic", "passing", 75)


    create_linear_regression_plot_overall_threshold(data,"age", "pace", 72)

    create_linear_regression_plot_release_clause(data,"age", 78)


if __name__ == "__main__":
    main()
