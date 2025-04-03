## Project Overview
This project analyzes the FIFA 22 player dataset, extracting statistical insights and visualizations. The main script processes the data, generates various statistical summaries, and creates multiple charts to visualize different player attributes.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- Required Python libraries:
  - pandas
  - seaborn
  - matplotlib
  
You can install the necessary dependencies using:
```sh
pip install -r requirements.txt
```

## Project Structure
```
FIFA22-Analysis/
│── basic_statistics.py         # Functions for numerical and categorical statistics
│── charts.py                   # Functions for creating charts
│── main.py                      # Main script to execute analysis
│── players_22.csv               # Dataset file
│── charts/                      # Directory for saved visualizations
│   ├── boxplots/
│   ├── error_bars/
│   ├── heatmaps/
│   ├── histograms/
│   ├── regression/
│   ├── violinplots/
│── README.md                    # Project documentation
```

## How to Run
1. Ensure the dataset file (`players_22.csv`) is in the project directory.
2. Run the main script:
```sh
python data_analysis_main.py
```

## Output
All generated visualizations will be saved in the `charts/` directory, categorized into subfolders based on the chart type.

## Author
Filip

