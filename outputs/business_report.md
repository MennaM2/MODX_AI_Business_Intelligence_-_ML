# Business Analysis Report

_Generated 2026-09-17 09:32:30_

## Executive Summary

- 'annual_income' has 76.25% missing values, which may bias any analysis using it.
- annual_income is increasing overall (26.13% total change).
- The sharpest drop in annual_income was -100.0% in 2023-02-28.
- annual_income moved from 3274000.0 in 2023-03 to 1868000.0 in 2023-07 (-42.94%).
- 'Germany' had the steepest decline (-96.1%) between the two periods.
- By annual_income, 'USA' leads (3398000.0) and 'France' trails (1015000.0) among country.
- A baseline model predicts 'country' with 0.225 accuracy; the top driver is 'numerical__internal_record_number'.

## Key Metrics

- Rows: 400
- Columns: 11
- Duplicate rows: 0
- Metric analyzed: annual_income
- Date column used: signup_date
- Grouping dimension: country

## Visual Overview

![Distribution of annual_income](outputs/histogram_annual_income.png)

![Correlation heatmap](outputs/correlation_heatmap.png)

## Trends

- Overall direction for **annual_income**: increasing
- Total change over the period: 26.13%
- Largest single-period drop: -100.0% in 2023-02-28
- Largest single-period rise: inf% in 2023-03-31

![annual_income trend over time](outputs/report_trend_chart.png)

## Period Comparison

- 2023-03: 3274000.0
- 2023-07: 1868000.0
- Change: -1406000.0 (-42.94%, decrease)

Fastest-growing country:

- France: 196.1%
- UK: 9.82%

Fastest-declining country:

- Germany: -96.1%
- USA: -41.03%

## Anomalies

- No significant anomalies detected.

## Top / Bottom Performers

Top country by annual_income (sum):

- USA: 3398000.0
- Germany: 2327000.0
- UK: 1280000.0
- France: 1015000.0

Bottom country by annual_income:

- France: 1015000.0
- UK: 1280000.0
- Germany: 2327000.0
- USA: 3398000.0

![Top and bottom country performers](outputs/report_top_bottom_chart.png)

## Predictive Signal

- Target column: country
- Task type: classification
- accuracy: 0.225
- precision: 0.2388
- recall: 0.225
- f1_score: 0.2229

Top features:
- numerical__internal_record_number: 0.4756
- numerical__age: 0.1373
- numerical__annual_income: 0.131
- categorical__signup_date_2023-07-22: 0.0275
- categorical__notes_vip: 0.0274

## Insights

- 'annual_income' has 76.25% missing values, which may bias any analysis using it.
- annual_income is increasing overall (26.13% total change).
- The sharpest drop in annual_income was -100.0% in 2023-02-28.
- annual_income moved from 3274000.0 in 2023-03 to 1868000.0 in 2023-07 (-42.94%).
- 'Germany' had the steepest decline (-96.1%) between the two periods.
- By annual_income, 'USA' leads (3398000.0) and 'France' trails (1015000.0) among country.
- A baseline model predicts 'country' with 0.225 accuracy; the top driver is 'numerical__internal_record_number'.

## Recommendations

- Investigate why 'annual_income' has a high missing rate before using it in reporting or modeling.
- Review what happened around 2023-02-28 - it had the largest period-over-period drop in annual_income.
- Prioritize investigating 'Germany' - it declined -96.1% between the two most recent periods.
- Examine why 'France' underperforms other country on annual_income, and whether 'USA' has practices worth replicating.

## Limitations

- Generated automatically from the uploaded CSV only; it does not incorporate context outside the data.
- Date, metric, and grouping columns were auto-detected when not specified and may not match true business intent.
- Correlations, trends, and feature importance reflect association within this dataset, not proven causation.