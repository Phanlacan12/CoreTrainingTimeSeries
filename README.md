<div align="center">
<img alt="Pipeline" src="pictures/logo/logo.png" width="100%"/>
</div>


# TSFB: PLA Time Series Forecasting Benchmark

## Table of Contents

1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Quickstart](#quickstart)
5. [Configuration](#configuration)
6. [Run pipeline](#run_pipeline)
7. [Approaches](#approaches)
8. [Evaluation Strategies](#evaluation-strategies)
9. [Use Case](#use-case)

## Introduction

TSFB (Time Series Forecasting Benchmark) is a versatile framework designed to streamline the development, evaluation, and comparison of forecasting methods across a wide range of time series applications. Whether you’re working on financial price prediction, energy load forecasting, epidemiological modeling, or any domain that produces sequential data, TSFB provides a standardized pipeline to:

- Ingest and preprocess raw time series and exogenous covariates (past, future, static).

- Define modeling approaches (univariate, multivariate, hybrid) with pluggable adapters for popular libraries (Darts, TS-Lib, custom implementations).

- Configure and train models via a unified YAML interface, leveraging CPU/GPU and distributed backends like Ray.

- Evaluate performance using both fixed and rolling forecasting strategies, with a suite of built-in metrics (MAE, MSE, SMAPE, etc.).

- Compare results across methods, datasets, and hyper-parameter settings in a reproducible manner.

Below is a visual overview of the TSFB pipeline, which maps each stage—from data loading through evaluation—to sections in the config.yaml file.


<div align="center">
<img alt="Pipeline" src="pictures/pipeline.png" width="80%"/>
</div>

The pipeline is organized into four conceptual layers, each corresponding to specific sections in `config.yaml`:

1. **Data Layer**: Handles data ingestion and splitting into time series and covariates (past, future, static).

   * **Config section**: `data_module_config`
2. **Approach Layer**: Defines how forecasting approaches are generated (univariate, multivariate, or hybrid).

   * **Config section**: `model.*.approach` and `model.*.group`
3. **Model Layer**: Specifies the underlying forecasting library adapters and hyper-parameters.

   * **Config section**: `model.*.model_name` and `model.*.model_hyper_params`
4. **Evaluation Layer**: Selects evaluation metrics and strategies for forecasting.

   * **Config section**: `evaluation.strategy_args` and `evaluation.metrics`

Each layer flows into the next, forming a clear separation of concerns:

| **Layer**            | **Description**                                                            | **Config Section (YAML)**                                                   |
| -------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Data Layer**       | - Ingest raw series and covariates<br/>- Preprocess (impute, normalize)<br/>- Split into train/val/test | `data_module_config`                                                         |
| **Approach Layer**   | - Define forecasting tasks (univariate, multivariate, hybrid)              | `model.<approach_name>.approach`<br/>`model.<approach_name>.group`           |
| **Model Layer**      | - Instantiate model adapters<br/>- Configure hyperparameters               | `model.<approach_name>.model_name`<br/>`model.<approach_name>.model_hyper_params` |
| **Evaluation Layer** | - Execute forecasts (fixed or rolling)<br/>- Compute performance metrics    | `evaluation.strategy_args`<br/>`evaluation.metrics`                          |

With this mapping, users can easily navigate between the high-level pipeline diagram and the detailed YAML configuration below.


## Setup

### Requirements

* Python 3.11

### Data

Download the dataset from the Survey data sheet in: [Survey data](https://docs.google.com/spreadsheets/d/18SzcMZmQl0JnJS1J1LGin38zWxdG1ABGkL25UiiP468/edit?usp=sharing)

## Quickstart

### Installation

Clone the repository and pip install this package:

   ```bash
   git clone https://github.com/vnpt-ds-core/forecasting.git
   cd forecasting
   pip install -e .
   ```

or if you dont want to clone the repo, you can install directly via pip:

   ```bash
   pip install git+https://github.com/vnpt-ds-core/forecasting.git
   ```

### Run pipeline
- Run end to end pipeline with a specified config file:
   ```bash
    from tsfb import PipelineExecutor

    pipeline = PipelineExecutor(config_path="./tsfb_configs/use_case.yaml")
    pipeline.run()
   ```


- Run end to end pipeline and select the best model based on a specified metric (e.g., MAE):
   ```bash
    from tsfb import PipelineExecutor

    pipeline = PipelineExecutor(config_path="./tsfb_configs/use_case.yaml")
    pipeline.run_and_select_best(metric="mae", mode="auto")

   ```


### Configuration

The TSFB pipeline is driven entirely by the `config.yaml` file, which is structured into three main sections:

1. **data\_loader\_config**: Defines how to load and preprocess your time series data.
2. **backend**: Specifies the execution environment (e.g., CPU, GPU, Ray).
3. **model**: Describes one or more forecasting approaches and their hyper-parameters.
4. **evaluation**: Controls the forecasting horizon, strategy, and metrics.

Below is a detailed breakdown of each section:

```yaml
# 1) Data Loader Configuration
data_module_config: &base_loader    # Anchor for reuse in evaluation
  loader: base                     # Loader type: 'base' for single-file CSV, or your custom loader
  file_path: "./data/ETTh1.csv"  # Path to your dataset
  index_col: date                  # Column name representing the datetime index
  target_columns:                  # List of columns to forecast
    - HUFL
    - MUFL
    - HULL
    - LULL
  covariate:
    past:
      - mobility_inflow      # number of people returning from high-risk areas
    future:
      - day_of_week         # day of week (0=Monday…6=Sunday)
      - is_holiday          # public holiday flag
      - influenza_visits    # clinic visits for influenza
      - h5n1_visits         # clinic visits for H5N1
      - covid_visits        # clinic visits for COVID-19
    static:
      - commune
  normalize:
    method: zscore
  split_ratio:
    train: 0.6
    val: 0.2
    test: 0.2
  return_type: pandas              # Output type for loaded data: 'pandas' or 'darts'

# 2) Execution Backend
backend:
  type: ray                        # Engine: 'ray', 'multiprocessing', or 'none'
  gpu_devices:                     # List of GPU IDs if using GPU
    - 0

# 3) Model Definitions
model:
  # Single (base) approach example
  base: # <-- This is name of a approach, will show up in approach_name in result csv file; not a base class
    approach: univariate_per_series # Choose one of: univariate_per_series, multivariate, hybrid
    series:                         # (For univariate or multivariate)
      - HUFL
      - MUFL                        # In multivariate, list all channels together
    model_name: "darts.models.NLinearModel"
    model_hyper_params:
      input_chunk_length: 128       # Number of past steps fed to the model
      output_chunk_length: 96       # Forecast horizon
      n_epochs: 1                   # Number of training epochs
      batch_size: 32
      random_state: 42
      norm: false                   # Whether to apply internal normalization
      pl_trainer_kwargs:            # PyTorch Lightning Trainer args
        accelerator: gpu
        devices: [0]
        enable_progress_bar: true

  # Hybrid approach example
  hybrid: # <-- This is name of a approach, will show up in approach_name in result csv file; not a base class
    approach: hybrid
    group:
      - series: ["HUFL", "MUFL"]
        sub_approach: univariate_per_series
      - series: ["HULL", "LULL"]
        sub_approach: multivariate

# 4) Evaluation Settings
evaluation:
  horizon: 96                       # Forecast length for evaluation
  series_list: ["__default__"]     # Which series groups to evaluate
  strategy_args:
    strategy_name: fixed_forecast   # 'fixed_forecast' or 'rolling_forecast'
    seed: 42                        # Random seed for reproducibility
    deterministic: full             # 'full' to enforce deterministic behavior
    stride: 1                       # (For rolling) step size between windows
    num_rollings: 1000              # (For rolling) total forecast iterations
    data_module_config: *base_loader# Reuse the data_module_config anchor
    save_true_pred: false           # Whether to store predictions and true values
    target_channel:
      "__default__": null         # Map series groups to target channels
  metrics: all                      # 'all' or a list of metrics (mae, mse, etc.)
```

### GPU and CPU configuration
#### GPU Configuration
When running Deep Learning models, we can use the GPU to accelerate the training process. To use the GPU, we need to configure the `config.yaml` file as follows:


```yaml
backend:
  type: ray
  gpu_devices: [0] # <-- Assign GPU configuration for backend
```
Next, for models using PyTorch in general and Darts specifically, we continue configuring the model_hyper_params section as follows:
```yaml
model:
  single_model:
    approach: default
    model_name: "darts.models.NLinearModel"
    model_hyper_params:
      input_chunk_length: 24
      output_chunk_length: 24
      n_epochs: 1
      batch_size: 32
      random_state: 42
      pl_trainer_kwargs:
        accelerator: gpu # <-- Run on GPU
        devices: [0] # <-- Choose which GPU to use
```
#### CPU Configuration
If we are using the CPU, we can configure it as follows:
```yaml
backend:
  type: ray
  gpu_devices: [] # <-- Do not assign GPU to the backend
```
Then, for models using PyTorch in general and Darts specifically, we continue configuring the model_hyper_params section as follows:
```yaml
model:
  single_model:
    approach: default
    model_name: "darts.models.NLinearModel"
    model_hyper_params:
      input_chunk_length: 24
      output_chunk_length: 24
      n_epochs: 1
      batch_size: 32
      random_state: 42
      pl_trainer_kwargs:
        accelerator: cpu # <-- Run on CPU
#        devices: [0] # <-- Do not set devices if not using GPU
```

#### Tips

* Use the `&base_loader` anchor to avoid duplication when referencing `data_module_config` in multiple places.

* For **hybrid** models, define each group with its own `sub_approach`, and TSFB will launch separate pipelines.

* Customize `split_ratio` based on your seasonality and sample size to ensure robust validation.

* When using **rolling\_forecast**, adjust `stride` and `num_rollings` to balance evaluation granularity and compute time.

* **Covariate**: Exogenous variables that support forecasting but are not target values.

  * **Past covariates**: Observed values available up to the forecast start.
  * **Future covariates**: Known or forecasted values for the prediction horizon (e.g., holidays, weather forecasts).
  * **Static covariates**: Time-invariant features describing each series (e.g., geographic region, product category).

* You can find how to configure the Darts model in the config files located in the `tsfb_configs` folder.

## Approaches

TSFB supports three main modeling approaches:

### 1. `univariate_per_series`

* **Concept**: Train a separate univariate model for each time series independently.
* **Pros**:

  * Hyper-parameters can be tuned specifically for each series.
  * Easier to debug and interpret individual models.
* **Cons**:

  * Does not leverage cross-series information.
  * Computational cost scales linearly with the number of series.
* **Use case**: Use when series are largely uncorrelated or require specialized configurations.
* **Example**: Forecast daily call volume for each province separately.

```yaml
model:
  per_province: # <-- This is name of a approach, will show up in approach_name in result csv file; not a base class
    sub_approach: univariate_per_series
    series: ["CallVolume_Hanoi"]
    model_name: "darts.models.NLinearModel"
    model_hyper_params:
      input_chunk_length: 168
      output_chunk_length: 24
      n_epochs: 10
```

---

### 2. `multivariate`

* **Concept**: Train a single multivariate model that consumes multiple series simultaneously.
* **Pros**:

  * Captures inter-series correlations and shared seasonality.
  * Reduces the number of separate models to maintain.
* **Cons**:

  * Can overfit if data is limited.
  * Higher resource requirements.
* **Use case**: Use when series exhibit strong relationships.
* **Example**: Forecast data volume and call volume together for a single city.

```yaml
model:
  city_services: # <-- This is name of a approach, will show up in approach_name in result csv file; not a base class
    sub_approach: multivariate
    series: ["DataVolume_HCM", "CallVolume_HCM"]
    model_name: "darts.models.NLinearModel"
    model_hyper_params:
      input_chunk_length: 168
      output_chunk_length: 24
      n_epochs: 20
```

---

### 3. `hybrid`

* **Concept**: Combine univariate and multivariate approaches by grouping series based on their relationships.
* **Pros**:

  * Flexible: leverage correlations where they exist and specialize where needed.
* **Cons**:

  * More complex configuration and debugging.
* **Use case**: Heterogeneous datasets where some series are correlated and others are independent.
* **Example**: Group data and call volume multivariately, and forecast other provinces independently.

```yaml
model:
  hybrid_policy: # <-- This is name of a approach, will show up in approach_name in result csv file; not a base class
    group:
      - series: ["DataVolume_HCM", "CallVolume_HCM"]
        sub_approach: multivariate
      - series: ["CallVolume_DaNang", "CallVolume_CanTho"]
        sub_approach: univariate_per_series
```

## Evaluation Strategies

TSFB includes two evaluation strategies:

### `fixed_forecast`

* **Description**: Perform a single-shot forecast for a fixed horizon.
* **Pros**: Fast and simple; good for point-in-time comparisons.
* **Cons**: Does not assess stability over time.
* **Use case**: One-off performance reports.
* **Example**: Forecast next month's revenue.

```yaml
evaluation:
  strategy_args:
    strategy_name: fixed_forecast
    data_module_config: *base_loader
    horizon: 96
```

---

### `rolling_forecast`

* **Description**: Perform consecutive forecasts by sliding or expanding the training window.
* **Parameters**:

  * `stride`: Time steps between consecutive forecasts.
  * `num_rollings`: Number of forecasting iterations.
  * `window_type`: `expanding` or `sliding` (optional).
* **Pros**: Evaluates model stability and adaptability; useful for drift detection.
* **Use case**: Continuous monitoring in production.
* **Example**: Daily data volume forecasts over 30 days.

```yaml
evaluation:
  strategy_args:
    strategy_name: rolling_forecast
    data_module_config: *base_loader
    stride: 1
    num_rollings: 30
    horizon: 24
```

## Use Case and Pipeline Mapping

### Case Study 1: Disease Forecasting

**Problem Statement:**
We have a dataset with daily records for multiple communes containing:

* `date`: Observation date.
* `commune`: Commune name (static covariate).
* **Disease case counts** (target columns):

  * `influenza_cases` (daily influenza cases)
  * `h5n1_cases` (daily H5N1 cases)
  * `covid_cases` (daily COVID-19 cases)
* **Clinic visit counts** (future covariates known ahead):

  * `influenza_visits` (number of influenza-related visits)
  * `h5n1_visits` (number of H5N1-related visits)
  * `covid_visits` (number of COVID-19–related visits)
* **Past covariates** derived from history:

  * `previous_day_cases` (lag-1 of each disease case count)
  * `rolling_avg_cases_7d` (7-day rolling average of cases)
  * `mobility_inflow` (daily count of people returning from high-risk areas)

The objective is to forecast the next 7 days of each disease’s case counts per commune, leveraging both historical case patterns and exogenous signals.


Before writing the YAML, we first analyze our disease‑forecasting scenario and define guiding questions for each pipeline layer.

#### Analysis

1. **Objective**: Predict the next 7 days of influenza, H5N1, and COVID-19 case counts across multiple communes.
2. **Data Requirements**:

   * What is the **time granularity** (daily)?
   * Which columns represent **targets** (case counts)?
   * What **static features** (commune) describe each series?
   * Which **exogenous signals** are available as:
     • Future-known (e.g., `day_of_week`, `is_holiday`, clinic visits)
     • Past-derived (e.g., lag features, rolling averages, mobility inflow)
3. **Approach Selection**:

   * Would using a single multivariate model to forecast 3 series be more effective than splitting them into multiple models? --> Lets try it all.
   * Do diseases **co-move** in the same commune? If yes, consider **multivariate**.
   * Are there communes with **sparse** or **independent** patterns? Use **univariate\_per\_series**.
   * Combine both via a **hybrid** group configuration.
4. **Model Configuration**:

   * How many past days (`input_chunk_length`) are needed to capture seasonality? (e.g., 28 days)
   * What is the forecast horizon (`output_chunk_length`)? (7 days)
   * Resource constraints: CPU/GPU, batch size, training epochs.
5. **Evaluation Strategy**:

   * Single-shot vs. rolling evaluation: Do we need to monitor **stability** over time? Choose `rolling_forecast` with appropriate `stride` and `num_rollings`.
   * Which **metrics** quantify error and shape? (e.g., MAE, SMAPE)

---

### Detailed Mapping to Pipeline Layers

Below is the step-by-step YAML for our use case, annotated with the guiding questions:

#### 1. Data Layer

```yaml
# Q1: What file holds our time series and covariates?
data_module_config: &disease_loader
  loader: base
  file_path: "./data/commune_disease_data.csv"
  # Q2: Which column is the datetime index?
  index_col: date

  # Q3: What are our target series?
  target_columns:
    - influenza_cases
    - h5n1_cases
    - covid_cases

  covariate:
    # Q4: Which features require lag or aggregation? (past)
    past:
      - previous_day_cases
      - rolling_avg_cases_7d
      - mobility_inflow
    # Q5: Which signals are known ahead? (future)
    future:
      - day_of_week
      - is_holiday
      - influenza_visits
      - h5n1_visits
      - covid_visits
    # Q6: Which static identifier describes each series?
    static:
      - commune

  # Q7: How to normalize and split data for train/val/test?
  normalize:
    method: zscore
  split_ratio:
    train: 0.6
    val: 0.2
    test: 0.2
  return_type: pandas
```

#### 2. Approach Layer

```yaml
# Q8: Would using a single multivariate model to forecast 3 series be more effective than splitting them into multiple models?
model:
  disease_hybrid: # <-- This is name of a approach, will show up in approach_name in result csv file; not a class
    approach: hybrid # Combine univariate and multivariate (1 model univariare for 2 series, 1 model multivariate for influenza_cases series)
    group:
      # Multivariate group for strong co-movement
      - series: ["h5n1_cases", "covid_cases"]
        sub_approach: multivariate
      # Univariate group for independent patterns
      - series: ["influenza_cases"]
        sub_approach: univariate_per_series

  disease_univariate: # <-- This is name of a approach, will show up in approach_name in result csv file; not a class
    approach: univariate_per_series # Loop 3 model univariate for each series
    series: ["influenza_cases", "h5n1_cases", "covid_cases"]

  disease_default: # <-- This is name of a approach, will show up in approach_name in result csv file; not a class
    approach: default # Use single multivariate model for all series
    series: ["influenza_cases", "h5n1_cases", "covid_cases"]
```

#### 3. Model Layer

```yaml
# Q9: What model class and hyper-parameters suit our task?
model:
  disease_hybrid:
    group:
      - sub_approach: multivariate
        model_name: "darts.models.NLinearModel"
        model_hyper_params:
          input_chunk_length: 28
          output_chunk_length: 7
          n_epochs: 20
          batch_size: 64
          random_state: 2025
      - sub_approach: univariate_per_series
        model_name: "darts.models.NLinearModel"
        model_hyper_params:
          input_chunk_length: 28
          output_chunk_length: 7
          n_epochs: 10
          batch_size: 32
          random_state: 2025
```

#### 4. Evaluation Layer

```yaml
# Q10: How do we validate performance over time?
evaluation:
  horizon: 7
  series_list: ["__default__"]
  strategy_args:
    strategy_name: rolling_forecast
    seed: 2025
    deterministic: full
    stride: 7           # forecast weekly
    num_rollings: 4     # four forecasts
    data_module_config: *disease_loader
    save_true_pred: true
    target_channel:
      "__default__": null
  metrics:
    - mae
    - smape
```

### Case Study 2: Retail Store Revenue Forecasting

**Problem Statement:**
We have a daily dataset for multiple retail stores, including:

- `date`: Observation date.
- `store_id`: Store identifier (static covariate).
- **Targets:**
  - `daily_revenue` (daily revenue)
  - `num_transactions` (daily number of transactions)
- **Future covariates** (known in advance):
  - `is_promotion` (promotion flag for that day)
  - `marketing_spend` (daily marketing budget)
  - `is_holiday` (holiday flag)
- **Past covariates** (derived from history):
  - `lag_1_revenue` (revenue lagged by 1 day)
  - `rolling_avg_revenue_7d` (7-day rolling average of revenue)

The objective is to forecast the next 14 days of daily revenue and transaction counts for each store, leveraging both historical trends and exogenous signals.

#### Analysis

1. **Objective**
   Predict the next 14 days of `daily_revenue` and `num_transactions` for each store.

2. **Data Requirements**
   - **Time granularity:** Daily
   - **Target columns:** `daily_revenue`, `num_transactions`
   - **Static covariates:** `store_id` (optionally add `store_type`, `region`)
   - **Other covariates:**
     - **Future-known:** `is_promotion`, `marketing_spend`, `is_holiday`
     - **Past-derived:** `lag_1_revenue`, `rolling_avg_revenue_7d`, `foot_traffic`

3. **Approach Selection**
   - Use a single multivariate model to forecast both series simultaneously?
   - Or split by store (global vs. per-store models)?

4. **Model Configuration**
   - **History window** (`input_chunk_length`): e.g., 60 days
   - **Forecast horizon** (`output_chunk_length`): 14 days
   - Resources: CPU/GPU, batch size, number of epochs

5. **Evaluation Strategy**
   - Rolling vs. fixed forecasting? Monitor stability over time → choose `rolling_forecast` with appropriate `stride` and `num_rollings`.
   - Metrics: MAE, SMAPE

---

### Detailed Mapping to Pipeline Layers

#### 1. Data Layer

```yaml
data_module_config: &retail_loader
  # Q1: Which file contains the time series and covariates?
  loader: base
  file_path: "./data/store_revenue.csv"

  # Q2: Which column is the datetime index?
  index_col: date

  # Q3: What are our target series?
  target_columns:
    - daily_revenue
    - num_transactions

  covariate:
    # Q4: Which past-derived features do we need?
    past:
      - lag_1_revenue
      - rolling_avg_revenue_7d
      - foot_traffic
    # Q5: Which future-known signals do we include?
    future:
      - is_promotion
      - marketing_spend
      - is_holiday
    # Q6: Which static features describe each series?
    static:
      - store_id

  # Q7: How to normalize and split?
  normalize:
    method: zscore
  split_ratio:
    train: 0.7
    val: 0.15
    test: 0.15
  return_type: pandas
```

#### 2. Approach Layer

```yaml
approach:
  retail_global:
    # Q8: Single multivariate model for all stores?
    approach: default
    series: ["daily_revenue", "num_transactions"]

  retail_per_store:
    # Q9: Separate univariate models per store?
    approach: univariate_per_series
```

#### 3. Model Layer

```yaml
model:
  retail_global:
    # Q11: Which model class for the global approach?
    group:
      model_name: "darts.models.TFTModel"
      model_hyper_params:
        # Q12: History window length
        input_chunk_length: 60
        # Q13: Forecast horizon
        output_chunk_length: 14
        # Q14: Training epochs
        n_epochs: 30
        # Q15: Batch size
        batch_size: 128
        random_state: 2025
```

#### 4. Evaluation Layer

```yaml
evaluation:
  horizon: 14
  series_list: ["__default__"]
  strategy_args:
    strategy_name: rolling_forecast
    seed: 2025
    deterministic: full
    # stride=1 → daily rolling; num_rollings=6 → six forecast windows
    stride: 1
    num_rollings: 6
    data_module_config: *retail_loader
    save_true_pred: true
    target_channel:
      "__default__": null
  metrics:
    - mae
    - smape
```



This mapping ensures that each question from our use case analysis directly corresponds to a block in the YAML configuration, tightly coupling the scenario requirements with the TSFB pipeline.
