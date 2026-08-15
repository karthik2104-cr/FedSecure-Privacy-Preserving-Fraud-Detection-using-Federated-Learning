#FedSecure: A Privacy Preserving Fraud Detection using Federated Learning

This project contains a research-style Streamlit dashboard for multiclass fraud detection using **synthetic data**.

## Covered fraud categories
- Credit Card Fraud
- Identity Theft
- Money Laundering
- Transaction Laundering
- UPI Fraud
- ATM Skimming Fraud
- Legitimate transactions

## Project structure
- `generate_synthetic_data.py` - builds the synthetic dataset with 20 features
- `train_models.py` - trains multiple algorithms, compares performance, and saves the best model
- `app.py` - Streamlit dashboard for EDA, benchmarking, and inference
- `fraud_config.py` - shared feature names, labels, and fraud signal maps
- `artifacts/` - saved model, metrics, and research plots
- `fed_train.py` - performs federated training using FedAvg and FedSecure

## Setup
```bash
pip install -r requirements.txt
python generate_synthetic_data.py --output dataset.csv
python train_models.py --dataset dataset.csv --output-dir artifacts
streamlit run app.py
```

## Features
The project uses 20 engineered signals spanning:
- Amount and timing behavior
- Geo-spatial anomalies
- Device and login anomalies
- Velocity and beneficiary risk
- Merchant and channel risk
- Card-present vs digital risk
- KYC, IP, and network graph signals

Each fraud class is designed with **at least 8 core discriminative signals** to satisfy research-style multiclass detection requirements.

## Notes
This is a synthetic research prototype and not a production AML/fraud monitoring platform.
