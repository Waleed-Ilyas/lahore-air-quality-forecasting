import numpy as np

from aqforecast import evaluate as E


def test_point_metrics_perfect_and_biased():
    y = np.array([10.0, 20.0, 30.0])
    perfect = E.point_metrics(y, y)
    assert perfect["mae"] == 0 and perfect["r2"] == 1
    biased = E.point_metrics(y, y + 5)
    assert biased["mae"] == 5 and biased["bias"] == 5


def test_alert_metrics_counts_unhealthy_hours():
    y = np.array([10, 60, 70, 20])
    pred = np.array([10, 60, 30, 20])
    m = E.alert_metrics(y, pred)
    assert m["alert_precision"] == 1.0 and m["alert_recall"] == 0.5


def test_interval_coverage():
    m = E.interval_metrics([1, 5, 9], [0, 0, 0], [2, 4, 10])
    assert round(m["coverage_pct"], 1) == 66.7
