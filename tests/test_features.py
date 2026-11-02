import numpy as np
import pandas as pd

from time_to_default.features import add_features, scope_filters


def raw_rows():
    return pd.DataFrame({
        "loan_status": ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid",
                        "Current", "Charged Off", "Current"],
        "application_type": ["Individual", "Individual", "Joint App", "Individual", "Individual"],
        "home_ownership": ["RENT", "RENT", "OWN", "ANY", "MORTGAGE"],
        "annual_inc": [60_000.0, 50_000.0, 70_000.0, 40_000.0, 0.0],
        "issue_d": pd.to_datetime(["2016-01-01"] * 5),
    })


def test_scope_filters_funnel_counts_each_step():
    kept, funnel = scope_filters(raw_rows())
    assert len(kept) == 1
    assert funnel["contratos"].tolist() == [5, 4, 3, 2, 1, 1]
    assert funnel["removidos"].sum() == 4


def test_pti_and_bands():
    df = pd.DataFrame({
        "int_rate": [10.0], "dti": [20.0], "installment": [500.0], "annual_inc": [60_000.0],
        "fico_range_low": [700.0], "fico_range_high": [704.0], "loan_amnt": [10_000.0],
        "grade": ["B"], "home_ownership": ["MORTGAGE"], "emp_length": ["10+ years"],
        "verification_status": ["Verified"], "purpose": ["credit_card"], "revol_util": [30.0],
        "inq_last_6mths": [0.0], "delinq_2yrs": [0.0], "total_rec_prncp": [0.0], "recoveries": [0.0], "collection_recovery_fee": [0.0], "issue_d": pd.to_datetime(["2015-06-01"]),
    })
    out = add_features(df)
    assert np.isclose(out.loc[0, "pti"], 10.0)  # 500 / 5.000 de renda mensal
    assert out.loc[0, "income_band"] == "60–80 mil"
    assert out.loc[0, "fico"] == 702
    assert out.loc[0, "home"] == "financiada"
    assert out.loc[0, "emp_years"] == 10
    assert out.loc[0, "vintage"] == 2015
