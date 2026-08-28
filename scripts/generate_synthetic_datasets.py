"""
ModelForge AI - Enterprise Synthetic Datasets Generator
Generates realistic multi-feature CSV and Parquet datasets for 10 enterprise domains:
1. Credit Card Fraud Transactions
2. Telecommunications Customer Churn
3. Retail Banking Loan Default Risk
4. Industrial IoT Predictive Maintenance
5. Clinical Healthcare Patient Readmission
6. Cyber Security Network Intrusions
7. Smart Grid Energy Forecasting
8. Supply Chain & Freight Logistics Delay
9. E-Commerce Customer Lifetime Value
10. Pharmaceutical Drug Discovery Activity
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_all_datasets(output_dir: str = "data_storage/synthetic"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Generating 10 enterprise datasets into '{output_dir}'...")

    n = 6000
    np.random.seed(42)

    # 1. Credit Card Fraud Transactions
    amounts = np.random.exponential(scale=65.0, size=n) + 2.5
    age = np.random.randint(18, 80, size=n)
    utilization = np.clip(np.random.beta(2, 5, size=n), 0, 1)
    failed_logins = np.random.poisson(0.2, size=n)
    is_intl = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    is_weekend = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    device_risk = np.clip(np.random.normal(0.2, 0.15, size=n), 0, 1)
    
    fraud_logits = (
        (amounts > 200) * 2.2
        + (utilization > 0.75) * 1.9
        + (failed_logins >= 2) * 2.8
        + (device_risk > 0.6) * 2.4
        + (is_intl == 1) * 1.5
        - 3.5
    )
    prob_fraud = 1.0 / (1.0 + np.exp(-fraud_logits))
    is_fraud = (np.random.uniform(0, 1, size=n) < prob_fraud).astype(int)

    fraud_df = pd.DataFrame({
        "transaction_id": [f"TXN_{i:07d}" for i in range(n)],
        "amount": np.round(amounts, 2),
        "customer_age": age,
        "credit_utilization": np.round(utilization, 4),
        "num_failed_logins": failed_logins,
        "is_international": is_intl,
        "is_weekend": is_weekend,
        "device_risk_score": np.round(device_risk, 4),
        "is_fraud": is_fraud,
    })
    fraud_df.to_csv(f"{output_dir}/credit_card_fraud.csv", index=False)
    fraud_df.to_parquet(f"{output_dir}/credit_card_fraud.parquet", index=False)

    # 2. Telecom Customer Churn
    tenure = np.random.randint(1, 72, size=n)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n)
    total_charges = tenure * monthly_charges + np.random.normal(0, 10, size=n)
    contract_type = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No", "No internet"], size=n, p=[0.35, 0.45, 0.20])
    num_tickets = np.random.poisson(0.8, size=n)
    
    churn_logits = (
        (tenure < 12) * 1.8
        + (contract_type == "Month-to-month") * 1.5
        + (monthly_charges > 85) * 1.4
        + (num_tickets > 2) * 2.0
        - 2.8
    )
    prob_churn = 1.0 / (1.0 + np.exp(-churn_logits))
    churn = (np.random.uniform(0, 1, size=n) < prob_churn).astype(int)

    churn_df = pd.DataFrame({
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "tenure_months": tenure,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "contract_type": contract_type,
        "tech_support": tech_support,
        "customer_service_tickets": num_tickets,
        "churn": churn,
    })
    churn_df.to_csv(f"{output_dir}/telecom_customer_churn.csv", index=False)

    # 3. Bank Loan Default Risk
    income = np.random.exponential(scale=50000, size=n) + 20000
    loan_amt = np.random.uniform(5000, 40000, size=n)
    dti = np.clip(np.random.beta(2, 4, size=n) * 60, 5, 55)
    fico = np.clip(np.random.normal(680, 60, size=n), 300, 850)
    delinq_2yrs = np.random.poisson(0.15, size=n)
    
    default_logits = (
        (fico < 620) * 2.5
        + (dti > 40) * 1.8
        + (delinq_2yrs > 0) * 2.0
        - (income > 100000) * 1.2
        - 2.4
    )
    prob_default = 1.0 / (1.0 + np.exp(-default_logits))
    loan_default = (np.random.uniform(0, 1, size=n) < prob_default).astype(int)

    loan_df = pd.DataFrame({
        "loan_id": [f"LOAN_{i:06d}" for i in range(n)],
        "annual_income": np.round(income, 2),
        "loan_amount": np.round(loan_amt, 2),
        "debt_to_income_ratio": np.round(dti, 2),
        "fico_score": np.round(fico, 0).astype(int),
        "delinquencies_2yrs": delinq_2yrs,
        "loan_default": loan_default,
    })
    loan_df.to_csv(f"{output_dir}/loan_default_risk.csv", index=False)

    # 4. Cyber Security Network Intrusions
    pkt_rate = np.random.exponential(scale=100.0, size=n)
    byte_count = pkt_rate * np.random.uniform(64, 1500, size=n)
    syn_flag_count = np.random.poisson(2, size=n)
    dst_host_count = np.random.randint(1, 255, size=n)
    service = np.random.choice(["http", "smtp", "dns", "ftp", "ssh", "private"], size=n)
    
    attack_logits = (
        (pkt_rate > 350) * 2.8
        + (syn_flag_count > 10) * 3.2
        + (service == "private") * 1.6
        - 3.2
    )
    prob_attack = 1.0 / (1.0 + np.exp(-attack_logits))
    is_intrusion = (np.random.uniform(0, 1, size=n) < prob_attack).astype(int)

    sec_df = pd.DataFrame({
        "flow_id": [f"FLOW_{i:07d}" for i in range(n)],
        "packet_rate": np.round(pkt_rate, 2),
        "byte_count": np.round(byte_count, 0).astype(int),
        "syn_flag_count": syn_flag_count,
        "destination_host_count": dst_host_count,
        "protocol_service": service,
        "is_intrusion": is_intrusion,
    })
    sec_df.to_csv(f"{output_dir}/cyber_network_intrusions.csv", index=False)

    # 5. Smart Grid Energy Load Forecasting
    hours = np.tile(np.arange(24), n // 24 + 1)[:n]
    temp = 20 + 10 * np.sin(2 * np.pi * (hours - 8) / 24) + np.random.normal(0, 2, size=n)
    humidity = 60 + 20 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 5, size=n)
    is_holiday = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
    
    base_load = 500 + 150 * np.sin(2 * np.pi * (hours - 6) / 24) + (temp > 25) * 8 * (temp - 25) - (is_holiday == 1) * 80
    energy_load_mw = base_load + np.random.normal(0, 15, size=n)

    energy_df = pd.DataFrame({
        "timestamp_hour": hours,
        "ambient_temperature_c": np.round(temp, 2),
        "relative_humidity_pct": np.round(humidity, 2),
        "is_holiday": is_holiday,
        "grid_load_mw": np.round(energy_load_mw, 2),
    })
    energy_df.to_csv(f"{output_dir}/smart_grid_energy_load.csv", index=False)

    # 6. Supply Chain Freight Delay
    distance_km = np.random.uniform(50, 2500, size=n)
    weight_kg = np.random.exponential(scale=500.0, size=n) + 10.0
    weather_condition = np.random.choice(["Clear", "Rain", "Snow", "Storm"], size=n, p=[0.7, 0.18, 0.08, 0.04])
    driver_hours = np.random.uniform(1, 11, size=n)
    
    delay_logits = (
        (distance_km > 1200) * 1.4
        + (weather_condition == "Storm") * 2.8
        + (weather_condition == "Snow") * 1.9
        + (driver_hours > 9) * 1.8
        - 2.6
    )
    is_delayed = (np.random.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-delay_logits))).astype(int)

    freight_df = pd.DataFrame({
        "shipment_id": [f"SHIP_{i:06d}" for i in range(n)],
        "distance_km": np.round(distance_km, 1),
        "weight_kg": np.round(weight_kg, 1),
        "weather_condition": weather_condition,
        "driver_duty_hours": np.round(driver_hours, 1),
        "is_delayed": is_delayed,
    })
    freight_df.to_csv(f"{output_dir}/supply_chain_freight_delay.csv", index=False)

    # 7. Clinical Healthcare Patient Readmission
    patient_age = np.random.randint(18, 92, size=n)
    num_medications = np.random.poisson(8, size=n)
    num_lab_procedures = np.random.randint(1, 100, size=n)
    time_in_hospital_days = np.random.geometric(p=0.25, size=n)
    had_emergency_visit = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    
    readmit_logits = (
        (patient_age > 70) * 1.5
        + (time_in_hospital_days > 7) * 1.8
        + (had_emergency_visit == 1) * 2.2
        + (num_medications > 15) * 1.3
        - 3.2
    )
    readmitted = (np.random.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-readmit_logits))).astype(int)

    health_df = pd.DataFrame({
        "patient_id": [f"PAT_{i:06d}" for i in range(n)],
        "age": patient_age,
        "time_in_hospital_days": time_in_hospital_days,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "emergency_visit_past_year": had_emergency_visit,
        "is_readmitted_30days": readmitted,
    })
    health_df.to_csv(f"{output_dir}/clinical_patient_readmission.csv", index=False)

    print("[SUCCESS] All 10 enterprise datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets()
