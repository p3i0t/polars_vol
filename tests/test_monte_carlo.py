import math

import polars as pl
import pytest

from polars_vol import black_scholes, monte_carlo

CASES = [
    # (strike, spot, vol, time_to_expiry, rate)
    (100.0, 100.0, 0.20, 1.0, 0.05),
    (120.0, 100.0, 0.30, 0.5, 0.05),
    (100.0, 50.0, 0.40, 2.0, 0.03),
    (90.0, 100.0, 0.15, 0.1, 0.0),
    (250.0, 250.0, 1.2, 0.007, 0.0425),
    (0.0001, 1.0, 0.8, 3.0, -0.02),
    (100.0, 100.0, 0.2, 0.0, 0.05),  # at expiry: intrinsic
    (90.0, 100.0, 0.2, 0.0, 0.05),  # ITM at expiry
    (100.0, 100.0, 0.0, 1.0, 0.05),  # zero vol -> 0.0
    (100.0, 0.0, 0.2, 1.0, 0.05),  # zero spot -> 0.0
    (0.0, 100.0, 0.2, 1.0, 0.05),  # zero strike -> 0.0
    (100.0, 100.0, 0.2, -1.0, 0.05),  # past expiry: intrinsic
]

LIVE = [0, 1, 2, 3, 4, 5]  # rows where the option is live (t > 0, positive inputs)
N_PATHS = 500_000

ARGS: tuple[str, str, str, str, str] = (
    "strike",
    "spot",
    "vol",
    "time_to_expiry",
    "rate",
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strike": [c[0] for c in CASES],
            "spot": [c[1] for c in CASES],
            "vol": [c[2] for c in CASES],
            "time_to_expiry": [c[3] for c in CASES],
            "rate": [c[4] for c in CASES],
        }
    )


def _mc(is_call, **kwargs) -> pl.Expr:
    return monte_carlo(*ARGS, is_call, **kwargs).struct.field("price")


def test_price_within_reported_standard_error():
    # the MC price must agree with the closed form within a few of its own
    # reported standard errors; anything else is bias, not noise
    for is_call in (True, False):
        df = _frame().with_columns(
            exact=black_scholes(*ARGS, pl.lit(is_call)),
            mc=_mc(pl.lit(is_call), n_paths=N_PATHS),
            se=monte_carlo(*ARGS, pl.lit(is_call), n_paths=N_PATHS).struct.field(
                "error"
            ),
        )
        for i in LIVE:
            gap = abs(df["mc"][i] - df["exact"][i])
            assert gap < 5.0 * df["se"][i] + 1e-9, (i, is_call, gap, df["se"][i])


def test_standard_error_is_payoff_std_over_sqrt_n():
    # the payoff of a live option has std on the order of spot * vol * sqrt(t),
    # so the reported error must be close to that divided by sqrt(n_paths)
    df = _frame().with_columns(
        se=monte_carlo(*ARGS, pl.lit(True), n_paths=N_PATHS).struct.field("error"),
    )
    for i in LIVE:
        _k, s, v, t, _r = CASES[i]
        scale = s * v * math.sqrt(t) / math.sqrt(N_PATHS)
        assert 0.1 * scale < df["se"][i] < 2.0 * scale, (i, df["se"][i], scale)


def test_put_call_parity_within_forward_mc_error():
    # per path, call_payoff - put_payoff = S_T - K, so the estimator satisfies
    # C - P = disc * (mean(S_T) - K). The sample mean of S_T is itself a
    # Monte Carlo estimate of the forward, so parity holds within its error:
    # a few times std(S_T) / sqrt(n) discounted.
    n = 10_000
    df = _frame().with_columns(
        call=_mc(pl.lit(True), n_paths=n),
        put=_mc(pl.lit(False), n_paths=n),
    )
    for i in LIVE:
        k, s, v, t, r = CASES[i]
        disc = math.exp(-r * t)
        expected = s - k * disc
        var_st = s * s * math.exp(2.0 * r * t) * (math.exp(v * v * t) - 1.0)
        tol = 4.0 * disc * math.sqrt(var_st / n)
        assert math.isclose(df["call"][i] - df["put"][i], expected, abs_tol=tol), i


def test_deterministic_for_seed():
    base = _frame()
    same_a = base.with_columns(mc=_mc(pl.lit(True), n_paths=20_000, seed=7))["mc"]
    same_b = base.with_columns(mc=_mc(pl.lit(True), n_paths=20_000, seed=7))["mc"]
    other = base.with_columns(mc=_mc(pl.lit(True), n_paths=20_000, seed=8))["mc"]
    assert same_a.equals(same_b)
    # a different seed is a different estimator: it must differ somewhere live
    assert any(same_a[i] != other[i] for i in LIVE)


def test_lazy_context_partitioning_is_irrelevant():
    lf = _frame().lazy().with_columns(mc=_mc(pl.lit(True), n_paths=20_000))
    eager = lf.collect()["mc"]
    # same expression evaluated slice-wise must give identical values
    idx = pl.arange(0, pl.len())
    sliced = pl.concat([lf.filter(idx < 6).collect(), lf.filter(idx >= 6).collect()])[
        "mc"
    ]
    assert eager.equals(sliced)


def test_degenerate_rows_follow_black_scholes_convention():
    df = _frame().with_columns(
        mc=_mc(pl.lit(True), n_paths=100),
        se=monte_carlo(*ARGS, pl.lit(True), n_paths=100).struct.field("error"),
        exact=black_scholes(*ARGS, pl.lit(True)),
    )
    for i in range(6, len(CASES)):
        assert df["mc"][i] == df["exact"][i], i
        assert df["se"][i] == 0.0  # deterministic rows carry zero MC error


def test_null_propagates():
    df = (
        pl.DataFrame(
            {
                "strike": [100.0, 100.0, None],
                "spot": [100.0, None, 100.0],
                "vol": [0.2, 0.2, 0.2],
                "time_to_expiry": [1.0, 1.0, 1.0],
                "rate": [0.05, 0.05, 0.05],
                "is_call": [True, True, None],
            }
        )
        .with_columns(result=monte_carlo(*ARGS, "is_call", n_paths=1_000))
        .unnest("result")
    )
    assert df["price"].to_list()[1:] == [None, None]
    assert df["error"].to_list()[1:] == [None, None]
    assert df["price"][0] is not None


def test_is_call_column_selects_per_row():
    df = (
        _frame()
        .with_columns(
            call=_mc(pl.lit(True), n_paths=10_000),
            put=_mc(pl.lit(False), n_paths=10_000),
            is_call=pl.Series([True] * 6 + [False] * 6),
        )
        .with_columns(mixed=_mc("is_call", n_paths=10_000))
    )
    for i, ic in enumerate(df["is_call"].to_list()):
        assert df["mixed"][i] == (df["call"][i] if ic else df["put"][i]), i


def test_error_shrinks_with_more_paths():
    # 16x the paths -> about 4x smaller error; the reported error is itself a
    # sample estimate, so the ratio only holds within its own sampling noise
    df = (
        _frame()
        .head(1)
        .with_columns(
            se1=monte_carlo(*ARGS, pl.lit(True), n_paths=25_000).struct.field("error"),
            se2=monte_carlo(*ARGS, pl.lit(True), n_paths=400_000).struct.field("error"),
        )
    )
    expected_ratio = math.sqrt(400_000 / 25_000)
    got_ratio = df["se1"][0] / df["se2"][0]
    assert math.isclose(got_ratio, expected_ratio, rel_tol=0.05)


def test_n_paths_must_be_positive():
    df = _frame().head(1)
    with pytest.raises(pl.exceptions.PolarsError):
        df.with_columns(mc=_mc(pl.lit(True), n_paths=0))
