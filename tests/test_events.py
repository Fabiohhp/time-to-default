import pandas as pd
import pytest

from time_to_default.events import (
    CENSORED,
    DEFAULT,
    PREPAY,
    build_survival_target,
    months_between,
)

OBS_END = pd.Timestamp("2019-03-01")


def make(rows):
    df = pd.DataFrame(rows, columns=["issue_d", "last_pymnt_d", "loan_status", "term"])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    df["last_pymnt_d"] = pd.to_datetime(df["last_pymnt_d"])
    return df


def test_months_between_crosses_year():
    a = pd.Series(pd.to_datetime(["2015-11-01"]))
    b = pd.Series(pd.to_datetime(["2016-02-01"]))
    assert months_between(a, b).iloc[0] == 3


def test_default_is_month_after_last_payment():
    out = build_survival_target(
        make([["2016-01-01", "2016-08-01", "Charged Off", " 36 months"]]), OBS_END
    )
    assert out.loc[0, "duration"] == 8
    assert out.loc[0, "event"] == DEFAULT


def test_default_without_any_payment_is_month_one():
    out = build_survival_target(make([["2016-01-01", None, "Charged Off", " 36 months"]]), OBS_END)
    assert out.loc[0, "duration"] == 1
    assert out.loc[0, "event"] == DEFAULT


def test_early_payoff_is_prepayment():
    out = build_survival_target(make([["2016-01-01", "2017-01-01", "Fully Paid", " 36 months"]]), OBS_END)
    assert out.loc[0, "duration"] == 12
    assert out.loc[0, "event"] == PREPAY


def test_payoff_at_maturity_is_censored_at_term():
    out = build_survival_target(make([["2014-01-01", "2017-03-01", "Fully Paid", " 36 months"]]), OBS_END)
    assert out.loc[0, "duration"] == 36
    assert out.loc[0, "event"] == CENSORED


def test_current_is_censored_at_observation_end():
    out = build_survival_target(make([["2018-03-01", "2019-02-01", "Current", " 60 months"]]), OBS_END)
    assert out.loc[0, "duration"] == 12
    assert out.loc[0, "event"] == CENSORED
    assert out.loc[0, "term_months"] == 60


def test_late_is_censored_at_last_payment():
    out = build_survival_target(
        make([["2018-01-01", "2018-10-01", "Late (31-120 days)", " 36 months"]]), OBS_END
    )
    assert out.loc[0, "duration"] == 9
    assert out.loc[0, "event"] == CENSORED


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        build_survival_target(make([["2016-01-01", "2016-08-01", "Weird", " 36 months"]]), OBS_END)


def test_events_after_observation_end_are_censored():
    out = build_survival_target(
        make([["2018-01-01", "2018-11-01", "Charged Off", " 36 months"],
              ["2018-01-01", "2018-12-01", "Fully Paid", " 36 months"],
              ["2018-01-01", "2018-06-01", "Charged Off", " 36 months"]]),
        pd.Timestamp("2018-10-01"),
    )
    assert out["duration"].tolist() == [9, 9, 6]
    assert out["event"].tolist() == [CENSORED, CENSORED, DEFAULT]


def test_issued_after_observation_end_has_zero_duration():
    out = build_survival_target(make([["2018-12-01", "2019-02-01", "Current", " 36 months"]]),
                                pd.Timestamp("2018-11-01"))
    assert out.loc[0, "duration"] == 0
