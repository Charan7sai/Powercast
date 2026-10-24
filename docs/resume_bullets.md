# Resume bullets (real numbers from the 2025-01-01 to 2026-10-25 walk-forward test)

- Engineered a daily-refreshing pipeline ingesting 8 years of hourly day-ahead prices and archived weather forecasts for 3 European bidding zones (NL, DE-LU, FR), with leakage tests enforcing D-1 12:00 information sets
- Forecast day-ahead power prices with quantile LightGBM and online conformal intervals, beating the LEAR benchmark by 5.5% MAE (rMAE 0.571; Diebold-Mariano p<0.001 in 2 of 3 zones) and lifting 80% interval coverage from 60% to 79.8%
- Backtested LP-optimised battery arbitrage on probabilistic forecasts over 663 days, capturing 93.9% of perfect-foresight profit vs 88.5% for naive forecasts; risk-aware strategy cut worst-day loss from EUR 29 to EUR 7
- Served forecasts via FastAPI in Docker and a Streamlit dashboard with drift monitoring, refreshed daily through GitHub Actions; stress-tested on the 2022 energy crisis
