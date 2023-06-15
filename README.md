# Salary Estimator

This repository contains the data and training scripts to fit the _Salary Estimator_ for software engineers and data scientists. To download the data, go to the `data` directory, where you can download the json files used in the project.

## Usage Instructions

Setup the environment, and install the requierments.
```
conda create -n "project-salary" python=3.11 ipython
conda activate project-salary
conda install -r requirements.txt
```

After downloading the data and running the notebook to generate `experimental_data.csv`, run the experiments, .g.:
```
python scripts/experiments.py scripts/example-config-exp.yml
```

## Data
The source data comes from several places. The salary survey data comes from [levels.fyi](https://www.levels.fyi/js/salaryData.json). While the _cost of living index 2022_ and _cost of living index by state 2022_ are from [Kaggle](https://www.kaggle.com/datasets/ankanhore545/cost-of-living-index-2022) and [World Population Review (WPR)](https://worldpopulationreview.com/state-rankings/cost-of-living-index-by-state) respectively. You can download the later two after signing in.

### Experimental data
| variable      | definition                                            | source                     |
| ------------- | ----------------------------------------------------- | -------------------------- |
| timestamp     | Survey time                                           | levels |
| company       | Name of the company                                   | levels |
| level         | Internal role level                                   | levels |
| title         | Job title                                             | levels |
| tyc           | Total yearly compensation (in thousands)              | levels |
| location      | Place of work, country or state                       | levels |           
| yoe           | Years of experience                                   | levels |
| yac           | Years at company                                      | levels |
| base          | Base salary (in thousands)                            | levels |
| equity        | Equity (in thousands)                                 | levels |
| bonus         | Bonus (in thousands)                                  | levels |
| gender        | Female/not-Female                                     | levels |
| coli          | Cost of living index                                  | Kaggle and WPR |       

## Methods and Results
The experimental section explores two problems: estimating the expectation and estimating the range

### Results: Regressing Expectation

|                       |    R2 |   MSE |   MAE |   MPL |   MAPE |   MEDAE | NAME                                    |
|:----------------------|------:|------:|------:|------:|-------:|--------:|:----------------------------------------|
| LinearRegression      | 0.402 | 0.27  | 0.403 | 0.202 |  0.083 |   0.331 | hot-busy-quetzal-of-sorcery             |
| Ridge                 | 0.402 | 0.27  | 0.403 | 0.202 |  0.083 |   0.331 | gigantic-mighty-ammonite-of-wholeness   |
| Lasso                 | 0.402 | 0.27  | 0.403 | 0.201 |  0.083 |   0.328 | curly-obedient-dingo-of-feminism        |
| ElasticNet            | 0.402 | 0.27  | 0.403 | 0.202 |  0.083 |   0.33  | glistening-arrogant-mushroom-of-passion |
| BayesianRidge         | 0.402 | 0.27  | 0.403 | 0.202 |  0.083 |   0.331 | flawless-guppy-of-sudden-will           |
| DecisionTreeRegressor | 0.55  | 0.203 | 0.351 | 0.176 |  0.072 |   0.287 | placid-prophetic-seahorse-of-endeavor   |
| RandomForestRegressor | 0.558 | 0.2   | 0.349 | 0.174 |  0.071 |   0.286 | benevolent-eggplant-bloodhound-of-pizza |

### Results: Range Estimation

#### 0.25 Percentile Results:
|                           |     R2 |   MSE |   MAE |   MPL |   MAPE |   MEDAE | NAME                                |
|:--------------------------|-------:|------:|------:|------:|-------:|--------:|:------------------------------------|
| QuantileRegressor         | -0.261 | 0.569 | 0.608 | 0.218 |  0.121 |   0.536 | fair-victorious-nuthatch-of-inquire |
| GradientBoostingRegressor |  0.223 | 0.351 | 0.476 | 0.16  |  0.092 |   0.408 | greedy-cocky-bison-of-rain          |

#### 0.75 Percentile Results:
|                           |     R2 |   MSE |   MAE |   MPL |   MAPE |   MEDAE | NAME                                |
|:--------------------------|-------:|------:|------:|------:|-------:|--------:|:------------------------------------|
| QuantileRegressor         | -0.395 | 0.63  | 0.595 | 0.192 |  0.133 |   0.463 | hot-kickass-guppy-of-essence        |
| GradientBoostingRegressor |  0.126 | 0.395 | 0.486 | 0.154 |  0.107 |   0.4   | misty-warm-nightingale-of-lightning |


