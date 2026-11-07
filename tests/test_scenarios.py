import pytest

from forecaster.scenarios import demean, describe, mixture_median, noise_scale_units, scenario_keys, scenario_probs


def test_keys():
    assert scenario_keys(["A"]) == ["A=YES", "A=NO"]
    assert scenario_keys(["A", "B"])[1] == "A=YES|B=NO"
    with pytest.raises(ValueError):
        scenario_keys([])


@pytest.mark.parametrize("coupling", ["comonotone", "independent"])
def test_probs_keep_marginals(coupling):
    p = {"A": 0.9, "B": 0.6, "C": 0.3}
    pr = scenario_probs(p, coupling)
    assert sum(pr.values()) == pytest.approx(1)
    for q in p:
        assert sum(v for k, v in pr.items() if f"{q}=YES" in k) == pytest.approx(p[q])


def test_comonotone_two():
    pr = scenario_probs({"H": 0.94, "S": 0.75})
    assert pr == pytest.approx({"H=YES|S=YES": 0.75, "H=YES|S=NO": 0.19, "H=NO|S=YES": 0.0, "H=NO|S=NO": 0.06})
    with pytest.raises(ValueError):
        scenario_probs({"H": 1.2})


def test_describe_demean_median():
    assert describe("A=YES|B=NO", {"A": "House"}) == "House: YES; B: NO"
    probs = {"x": 0.5, "y": 0.5}
    dm = demean({"x": 2.0, "y": 0.0}, probs)
    assert dm == {"x": 1.0, "y": -1.0}
    assert mixture_median(dm, probs, 4.0) == pytest.approx(0, abs=1e-8)
    assert mixture_median({"x": 0.7}, {"x": 1.0}, 4.0) == pytest.approx(0.7)


def test_noise_scale():
    from scipy import stats
    assert stats.t.ppf(0.75, 4) * noise_scale_units(4) == pytest.approx(1)
