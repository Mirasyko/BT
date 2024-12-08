# BT

## Introduction

This repository contains code for bachelor thesis. We are trying to solve allocation of energy in community energy sharing. With respect to segmentation of whole project we tried to segment this code into several directories. In a first part we are dealing with analysis of data given by company.

## First Part: Analysis

Data were divided into 2 parts. First part was a production of energy and in the second part consumption. These files are not present in the directory, we will only add final version of dataset.

### Data

Data contains production and consumption of 10 subjects in Beroun. Weather data were downloaded from [Czech Hydrometeorological institute](https://intranet.chmi.cz/historicka-data/pocasi/denni-data/Denni-data-dle-z.-123-1998-Sb#). This data were used for regression analysis present in `productionAndWeather.ipynb`.

### Results

- We arrived at a model consisting of `Lagged_Total`, `Temperature`, `DaylightMinutes`.
  - `Lagged_Total`    = lagged value of total production
  - `Temperature`     = average temperature of a day.
    - This data is not available in time of prediction, but can be approximated using meteorological prediction, because those tend to be correct in this time frame. 
  - `DaylightMinutes` = Difference between sunrise and sunset in minutes
- All of the parameters are significant (p-value < 0.05)
- Model has a $R^2$ equal to $0.6$ which tells us that $60\%$ of variation can be explained by the model
- We could also choose only model consisting of `DaylightMinutes` and `Lagged_Total`.
- Variables are correlated so we lose nice interpretation.
