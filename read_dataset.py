import pandas as pd

from basic_statistics import numerical_statistics, categorical_statistics


def read_data(file_path):
    return pd.read_csv(file_path, low_memory=False)


def main():
    file = "players_22.csv"
    data = read_data(file)

    numerical_statistics(data)

    categorical_statistics(data)


if __name__ == "__main__":
    main()