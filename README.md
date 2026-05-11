# Distribute-then-Select: Correlated Perturbation for Collusion-Resilient Mean Estimation under Local Privacy

This repository contains the official Python implementation of the DTS (Distribute-then-Select) mechanism.

## Repository Structure

* core.py: The central library containing the implementation of the DTS mechanism, DPDataGenerator for synthetic data generation, DataHandler for real-world dataset preprocessing, and various LDP baselines (Duchi, Piecewise, Hybrid, Laplace, and Gaussian).
* exp_synthetic.py: Main evaluation script for synthetic datasets. It benchmarks all mechanisms across five diverse distributions: Normal, Uniform, Exponential, Pareto, and Gaussian Mixture Models (GMM).
* exp_real_data.py: Evaluation script for real-world datasets, including NYC Green Taxi, ACS Income (PSAM), and SF Employee Compensation data.
* exp_delta_sensitivity.py: A dedicated script for parameter sensitivity analysis, focusing on how the privacy parameter delta impacts the Mean Squared Error (MSE) within the DTS framework.

## Installation and Requirements

The implementation is built on Python 3.9+. To install the necessary libraries, run the following command in your terminal:

pip install numpy pandas scipy tqdm pyarrow

## Datasets and Reproducibility

To reproduce the experimental results, please download the raw datasets from the official sources provided below and place them in your local directory :

### 1. NYC Green Taxi Trip Records
- Source: [TLC Trip Record Data Official Page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Direct Download: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2018-01.parquet
- Attribute used: trip_distance.

### 2. ACS Income (PUMS)
- Source: [U.S. Census Bureau ACS Microdata](https://www.census.gov/programs-surveys/acs/microdata.html)
- Direct Download: https://www2.census.gov/programs-surveys/acs/data/pums/2018/5-Year/csv_pca.zip
- Attribute used: PINCP (Total Personal Income) .

### 3. SF Employee Compensation
- Source: [San Francisco Open Data Portal](https://data.sfgov.org/City-Management-and-Ethics/Employee-Compensation/88g8-5mnd)
- Direct Download: https://data.sfgov.org/api/views/88g8-5mnd/rows.csv?accessType=DOWNLOAD
- Note on Versioning: Our experiments were conducted using a snapshot with n = 1,053,560 records. While the online dataset is updated periodically, the core statistical characteristics remain consistent.
- Attribute used: Total Compensation.

## Usage

Each experiment can be executed by running the corresponding script from the terminal:

python exp_synthetic.py
python exp_real_data.py
python exp_delta_sensitivity.py
