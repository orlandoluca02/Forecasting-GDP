
# GDP Forecasting: GDFM vs Econometric Benchmarks

This repository features a professional MATLAB framework designed for macroeconomic forecasting. The project specifically focuses on predicting GDP growth by leveraging high-dimensional datasets through factor models.

## Project Objective

The goal of this project is to perform h-step ahead forecasting of GDP (Gross Domestic Product). It evaluates whether complex dynamic factor models can outperform simpler univariate and multivariate econometric benchmarks in a simulated real-time environment.

### Models and Comparison
The framework performs a rigorous comparison between:
* GDFM (Generalized Dynamic Factor Model): The primary model, which extracts dynamic common components from a large panel of variables to capture the underlying signal of the economy.
* Static PCA: A benchmark that uses standard Principal Component Analysis to build a factor-based forecast without accounting for lead-lag relationships.
* VAR (Vector Autoregression): A multivariate benchmark where the additional variables are selected via a data-driven correlation ranking with the target's common component.
* AR(1) (Autoregressive): The standard univariate benchmark for time-series persistence.
* Random Walk (RW): The naive benchmark (no-change forecast).

## Project Structure

* data/: Contains the dataset processed_data copia.xlsx.
* script/: Contains the main forecasting script main_gdp.m.
* tools/: Directory for external libraries and functions.
    * gdfm copia/: Core GDFM estimation and forecasting functions.
    * ABC_crit_fast/: Alessi-Barigozzi-Capasso (2010) factor selection criterion.

## Citations and Acknowledgments

This project implements methodologies from the following seminal works:

### Generalized Dynamic Factor Model (GDFM)
* Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2005). "The Generalized Dynamic Factor Model: One-Sided Estimation and Forecasting." JASA.
* Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000). "The Generalized Dynamic Factor Model: Identification and Estimation." The Review of Economics and Statistics.

### ABC Factor Selection Criterion
* Alessi, L., Barigozzi, M., & Capasso, M. (2010). "Improved Performance of the Hellin and Liška Test for Determining the Number of Factors." Journal of Applied Econometrics.

### Data Source
* The dataset used in this framework is sourced from Prof. Matteo Barigozzi's official website.

## Methodology Summary

* Evaluation: Out-of-Sample (OOS) rolling-origin evaluation.
* Rolling Window: 80 quarters (fixed).
* Forecast Horizon: 8-step ahead (h=8).
* Evaluation Metric: Root Mean Square Error (RMSE) and Relative RMSE (benchmarked against AR1).
