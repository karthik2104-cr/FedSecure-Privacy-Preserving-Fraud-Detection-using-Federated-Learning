FEATURES = [
    'transaction_amount',
    'transaction_hour',
    'merchant_risk_score',
    'geo_distance_km',
    'device_change_score',
    'account_age_days',
    'failed_login_attempts_24h',
    'velocity_1h',
    'velocity_24h',
    'beneficiary_age_hours',
    'cash_ratio_30d',
    'international_flag',
    'card_present_flag',
    'atm_distance_from_home_km',
    'upi_collect_request_flag',
    'kyc_mismatch_score',
    'ip_risk_score',
    'description_entropy',
    'chargeback_history_count',
    'network_graph_risk',
]

TARGET = 'fraud_type'

CLASS_LABELS = [
    'legitimate',
    'credit_card_fraud',
    'identity_theft',
    'money_laundering',
    'transaction_laundering',
    'upi_fraud',
    'atm_skimming_fraud',
]

CLASS_DESCRIPTION = {
    'legitimate': 'Normal low-risk activity with stable behavior and low anomaly scores.',
    'credit_card_fraud': 'Card-not-present anomalies, high merchant risk, unusual geolocation, and elevated digital risk.',
    'identity_theft': 'Takeover behavior marked by device change, failed logins, KYC mismatch, and abnormal profile shifts.',
    'money_laundering': 'Structuring and layering patterns with elevated cash ratio, network risk, and high transaction value.',
    'transaction_laundering': 'Concealed merchant behavior with abnormal merchant risk, velocity, and hidden channel complexity.',
    'upi_fraud': 'UPI pull-request abuse, rapid transfers, risky devices, and account takeover signals.',
    'atm_skimming_fraud': 'Card-present ATM withdrawals far from home, unusual timing, and suspicious withdrawal bursts.',
}

FRAUD_SIGNAL_MAP = {
    'credit_card_fraud': [
        'transaction_amount', 'merchant_risk_score', 'geo_distance_km', 'device_change_score',
        'velocity_1h', 'international_flag', 'card_present_flag', 'ip_risk_score',
        'chargeback_history_count', 'network_graph_risk'
    ],
    'identity_theft': [
        'device_change_score', 'failed_login_attempts_24h', 'geo_distance_km', 'account_age_days',
        'beneficiary_age_hours', 'kyc_mismatch_score', 'ip_risk_score', 'velocity_24h',
        'transaction_amount', 'network_graph_risk'
    ],
    'money_laundering': [
        'transaction_amount', 'velocity_24h', 'cash_ratio_30d', 'beneficiary_age_hours',
        'network_graph_risk', 'description_entropy', 'international_flag', 'merchant_risk_score',
        'chargeback_history_count', 'geo_distance_km'
    ],
    'transaction_laundering': [
        'merchant_risk_score', 'transaction_amount', 'velocity_1h', 'velocity_24h',
        'description_entropy', 'network_graph_risk', 'international_flag', 'geo_distance_km',
        'ip_risk_score', 'card_present_flag'
    ],
    'upi_fraud': [
        'upi_collect_request_flag', 'failed_login_attempts_24h', 'velocity_1h', 'beneficiary_age_hours',
        'device_change_score', 'account_age_days', 'kyc_mismatch_score', 'ip_risk_score',
        'transaction_amount', 'network_graph_risk'
    ],
    'atm_skimming_fraud': [
        'card_present_flag', 'atm_distance_from_home_km', 'transaction_hour', 'velocity_1h',
        'transaction_amount', 'geo_distance_km', 'international_flag', 'merchant_risk_score',
        'chargeback_history_count', 'ip_risk_score'
    ],
}
