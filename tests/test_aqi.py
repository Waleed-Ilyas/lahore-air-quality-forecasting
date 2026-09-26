import numpy as np

from aqforecast import aqi


def test_category_boundaries():
    assert aqi.category_name(0) == "Good"
    assert aqi.category_name(9.0) == "Good"
    assert aqi.category_name(9.1) == "Moderate"
    assert aqi.category_name(35.4) == "Moderate"
    assert aqi.category_name(55.5) == "Unhealthy"
    assert aqi.category_name(500) == "Hazardous"


def test_alert_threshold():
    assert aqi.is_alert(55.5)
    assert not aqi.is_alert(55.4)
    assert aqi.is_alert(np.array([10, 60, 200])).tolist() == [False, True, True]


def test_category_index_is_monotone():
    values = np.linspace(0, 400, 800)
    assert (np.diff(aqi.category_index(values)) >= 0).all()
