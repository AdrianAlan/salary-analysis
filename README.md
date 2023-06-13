# Salary Estimator

This repository contains the data and training scripts to fit the _Salary Estimator_ for software engineers and data scientists. To download the data, go to the `data` directory, where you can download the json files used in the project.

## Data
The source data comes from several places. The salary survey data comes from [levels.fyi](https://www.levels.fyi/js/salaryData.json). While the _cost of living index 2022_ and _cost of living index by state 2022_ are from [Kaggle](https://www.kaggle.com/datasets/ankanhore545/cost-of-living-index-2022) and [World Population Review (WPR)](https://worldpopulationreview.com/state-rankings/cost-of-living-index-by-state) respectively.

#### Experimental data
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

##Methods and Results
The experimental section explores two problems: estimating the expectation and estimating the range

#### Expectation
  
| Method | MAE | MAPE | MEDAE | MSE | R2 |
| --------------- | ----------------------------- | -------------------------- | ------------- | ----------------------------- | -------------------------- |
|Linear Regression | 0.403 | 0.083 | 0.331 | 0.270 | 0.402 |
|Ridge Regression  | 0.403 | 0.083 | 0.331 | 0.270 | 0.402 |
|Lasso Regression          | 0.403 | 0.083 | 0.328 | 0.270 | 0.402 |
|Elastic Net Regression    | 0.403 | 0.083 | 0.330 | 0.270 | 0.402 |
|Bayesian Ridge Regression | 0.403 | 0.083 | 0.331 | 0.270 | 0.402 |
|Decision Tree Regression  | 0.351 | 0.072 | 0.287 | 0.230 | 0.550 |
|Random Forest Regression  | 0.349 | 0.071 | 0.286 | 0.200 | 0.558 |

#### Range

| Method | Quantile | MPL |
| ------ | -------- | --- |
| Quantile Regression | 0.25 | 0.218 |
| | 0.75 | 0.192 |
| Gradient Boosting Regression | 0.25 | 0.160 |
| | 0.75 | 0.154 |

##Usage Instructions
