import math

import polars as pl
import pytest

from polars_vol import (
    black_scholes,
    black_scholes_with_greeks,
    implied_vol,
    iv_and_greeks,
)

SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT_2)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) * INV_SQRT_2PI


def ref_price(k, s, v, t, r, is_call) -> float:
    if t <= 0.0:
        return max(s - k, 0.0) if is_call else max(k - s, 0.0)
    if s <= 0.0 or k <= 0.0 or v <= 0.0:
        return 0.0
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))
    d2 = d1 - v * math.sqrt(t)
    disc = math.exp(-r * t)
    if is_call:
        return s * norm_cdf(d1) - k * disc * norm_cdf(d2)
    return k * disc * norm_cdf(-d2) - s * norm_cdf(-d1)


def ref_greeks(
    k, s, v, t, r, is_call
) -> tuple[float, float, float, float, float, float]:
    if t <= 0.0:
        price = max(s - k, 0.0) if is_call else max(k - s, 0.0)
        delta = (1.0 if s > k else 0.0) if is_call else (-1.0 if k > s else 0.0)
        return price, delta, 0.0, 0.0, 0.0, 0.0
    if s <= 0.0 or k <= 0.0 or v <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sqrt_t = math.sqrt(t)
    vol_sqrt_t = v * sqrt_t
    d1 = (math.log(s / k) + (r + 0.5 * v * v) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    disc = math.exp(-r * t)
    pdf_d1 = norm_pdf(d1)
    if is_call:
        price = s * norm_cdf(d1) - k * disc * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = -s * pdf_d1 * v / (2.0 * sqrt_t) - r * k * disc * norm_cdf(d2)
        rho = k * t * disc * norm_cdf(d2)
    else:
        price = k * disc * norm_cdf(-d2) - s * norm_cdf(-d1)
        delta = norm_cdf(d1) - 1.0
        theta = -s * pdf_d1 * v / (2.0 * sqrt_t) + r * k * disc * norm_cdf(-d2)
        rho = -k * t * disc * norm_cdf(-d2)
    gamma = pdf_d1 / (s * vol_sqrt_t)
    vega = s * pdf_d1 * sqrt_t
    return price, delta, gamma, vega, theta, rho


CASES = [
    # (strike, spot, vol, time_to_expiry, rate)
    (100.0, 100.0, 0.20, 1.0, 0.05),
    (120.0, 100.0, 0.30, 0.5, 0.05),
    (100.0, 50.0, 0.40, 2.0, 0.03),
    (90.0, 100.0, 0.15, 0.1, 0.0),
    (250.0, 250.0, 1.2, 0.007, 0.0425),
    (0.0001, 1.0, 0.8, 3.0, -0.02),
    (100.0, 100.0, 0.2, 0.0, 0.05),  # at expiry: intrinsic
    (90.0, 100.0, 0.2, 0.0, 0.05),
    (100.0, 100.0, 0.0, 1.0, 0.05),  # zero vol -> 0.0
    (100.0, 0.0, 0.2, 1.0, 0.05),  # zero spot -> 0.0
    (0.0, 100.0, 0.2, 1.0, 0.05),  # zero strike -> 0.0
    (100.0, 100.0, 0.2, -1.0, 0.05),  # past expiry: intrinsic
]

GREEK_NAMES = ["model_price", "delta", "gamma", "vega", "theta", "rho"]

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


def _matches_reference(is_call: bool) -> None:
    df = _frame().with_columns(
        result=black_scholes(*ARGS, pl.lit(is_call)),
    )
    for case, got in zip(CASES, df["result"].to_list()):
        expected = ref_price(*case, is_call)
        # the Rust kernel's A&S 7.1.26 CDF is accurate to ~1.5e-7
        assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-4), (
            case,
            is_call,
            got,
            expected,
        )


def test_call_matches_reference():
    _matches_reference(is_call=True)


def test_put_matches_reference():
    _matches_reference(is_call=False)


def test_is_call_column_selects_per_row():
    df = (
        _frame()
        .with_columns(
            call=black_scholes(*ARGS, pl.lit(True)),
            put=black_scholes(*ARGS, pl.lit(False)),
            is_call=pl.Series([True] * 6 + [False] * 6),
        )
        .with_columns(mixed=black_scholes(*ARGS, "is_call"))
    )
    calls = df["call"].to_list()
    puts = df["put"].to_list()
    mixed = df["mixed"].to_list()
    for i, ic in enumerate(df["is_call"].to_list()):
        assert mixed[i] == (calls[i] if ic else puts[i]), i


def test_known_value_hull():
    # Hull, "Options, Futures and Other Derivatives": c1 = 10.4506, p1 = 5.5735
    df = _frame().with_columns(
        call=black_scholes(*ARGS, pl.lit(True)),
        put=black_scholes(*ARGS, pl.lit(False)),
    )
    assert math.isclose(df["call"][0], 10.450584, abs_tol=1e-4)
    assert math.isclose(df["put"][0], 5.573526, abs_tol=1e-4)


def test_degenerate_rows():
    df = _frame().with_columns(
        call=black_scholes(*ARGS, pl.lit(True)),
        put=black_scholes(*ARGS, pl.lit(False)),
    )
    assert df["call"][6] == 0.0 and df["put"][6] == 0.0  # ATM at expiry
    assert df["call"][7] == 10.0 and df["put"][7] == 0.0  # ITM at expiry
    assert df["call"][8] == 0.0 and df["put"][8] == 0.0  # zero vol -> 0.0
    assert df["call"][9] == 0.0 and df["put"][9] == 0.0  # zero spot -> 0.0
    assert df["call"][10] == 0.0 and df["put"][10] == 0.0  # zero strike -> 0.0
    assert df["call"][11] == 0.0 and df["put"][11] == 0.0  # past expiry (ATM)


def test_null_propagates():
    df = pl.DataFrame(
        {
            "strike": [100.0, 100.0, None],
            "spot": [100.0, None, 100.0],
            "vol": [0.2, 0.2, 0.2],
            "time_to_expiry": [1.0, 1.0, 1.0],
            "rate": [0.05, 0.05, 0.05],
            "is_call": [True, True, None],
        }
    ).with_columns(
        price=black_scholes(*ARGS, "is_call"),
    )
    got = df["price"].to_list()
    assert got[1:] == [None, None]  # null spot and null is_call
    assert math.isclose(
        got[0],
        ref_price(100.0, 100.0, 0.2, 1.0, 0.05, True),
        rel_tol=1e-6,
        abs_tol=1e-4,
    )


def test_broadcast_scalar_and_int_columns():
    df = pl.DataFrame(
        {
            "spot": [100.0, 105.0, 110.0],
            "time_to_expiry": [1, 1, 1],  # ints should be cast
        }
    ).with_columns(
        price=black_scholes(
            pl.lit(100.0),
            "spot",
            pl.lit(0.2),
            "time_to_expiry",
            pl.lit(0.05),
            pl.lit(True),
        ),
    )
    assert df["price"].dtype == pl.Float64
    for spot, got in zip(df["spot"].to_list(), df["price"].to_list()):
        assert math.isclose(
            got,
            ref_price(100.0, spot, 0.2, 1.0, 0.05, True),
            rel_tol=1e-6,
            abs_tol=1e-4,
        )


def test_is_call_wrong_dtype_raises():
    df = pl.DataFrame(
        {
            "strike": [100.0],
            "spot": [100.0],
            "vol": [0.2],
            "time_to_expiry": [1.0],
            "rate": [0.05],
            "is_call": [1],
        }
    )
    with pytest.raises(pl.exceptions.PolarsError):
        df.with_columns(price=black_scholes(*ARGS, "is_call"))


def test_lazy_context():
    lf = (
        _frame()
        .lazy()
        .with_columns(
            price=black_scholes(*ARGS, pl.lit(False)),
        )
    )
    got = lf.collect()["price"].to_list()[0]
    assert math.isclose(got, ref_price(*CASES[0], False), rel_tol=1e-6, abs_tol=1e-4)


def _greeks_frame(is_call) -> pl.DataFrame:
    return (
        _frame()
        .with_columns(
            greeks=black_scholes_with_greeks(*ARGS, is_call),
        )
        .unnest("greeks")
    )


def test_greeks_match_reference():
    for is_call in (True, False):
        df = _greeks_frame(pl.lit(is_call))
        for case, row in zip(CASES, df.iter_rows(named=True)):
            expected = ref_greeks(*case, is_call)
            for name, exp in zip(GREEK_NAMES, expected):
                # gamma/vega are exact (only exp); the others carry the
                # A&S 7.1.26 CDF error
                assert math.isclose(row[name], exp, rel_tol=1e-6, abs_tol=1e-4), (
                    case,
                    is_call,
                    name,
                    row[name],
                    exp,
                )


def test_greeks_price_matches_black_scholes():
    df = (
        _frame()
        .with_columns(
            px=black_scholes(*ARGS, pl.lit(True)),
            greeks=black_scholes_with_greeks(*ARGS, pl.lit(True)),
        )
        .unnest("greeks")
    )
    assert (df["px"] == df["model_price"]).all()


def test_greeks_is_call_column_matches_scalar_calls():
    df = _greeks_frame(pl.lit(True)).drop(
        "model_price", "delta", "gamma", "vega", "theta", "rho"
    )
    call = df.with_columns(
        greeks=black_scholes_with_greeks(*ARGS, pl.lit(True))
    ).unnest("greeks")
    put = df.with_columns(
        greeks=black_scholes_with_greeks(*ARGS, pl.lit(False))
    ).unnest("greeks")
    mixed = df.with_columns(
        greeks=black_scholes_with_greeks(*ARGS, pl.Series([True] * 6 + [False] * 6))
    ).unnest("greeks")
    for name in GREEK_NAMES:
        expected = pl.concat([call[name].head(6), put[name].tail(6)])
        assert mixed[name].equals(expected), name


def test_delta_parity_call_minus_put_is_one():
    df = _frame().with_columns(
        c_delta=black_scholes_with_greeks(*ARGS, pl.lit(True)).struct.field("delta"),
        p_delta=black_scholes_with_greeks(*ARGS, pl.lit(False)).struct.field("delta"),
    )
    # only the live-option rows (t > 0, spot > 0, strike > 0, vol > 0) obey parity
    live = [0, 1, 2, 3, 4, 5]
    assert all(
        math.isclose(df["c_delta"][i] - df["p_delta"][i], 1.0, abs_tol=1e-6)
        for i in live
    )


def test_greeks_degenerate_rows():
    df = _greeks_frame(pl.lit(True))
    assert df["delta"][7] == 1.0  # ITM at expiry -> exercise indicator
    assert df["gamma"][7] == 0.0 and df["vega"][7] == 0.0
    assert df["delta"][6] == 0.0  # ATM at expiry
    for name in GREEK_NAMES:  # zero vol / spot / strike -> all 0.0
        for i in (8, 9, 10):
            assert df[name][i] == 0.0, (name, i)


def test_greeks_nulls_propagate():
    df = (
        pl.DataFrame(
            {
                "strike": [100.0, None],
                "spot": [100.0, 100.0],
                "vol": [0.2, 0.2],
                "time_to_expiry": [1.0, 1.0],
                "rate": [0.05, 0.05],
                "is_call": [True, True],
            }
        )
        .with_columns(
            greeks=black_scholes_with_greeks(*ARGS, "is_call"),
        )
        .unnest("greeks")
    )
    assert all(df[name].to_list()[1] is None for name in GREEK_NAMES)


# ------------------------------------------------------------------ implied vol

IV_ARGS: tuple[str, str, str, str, str, str] = (
    "option_price",
    "strike",
    "spot",
    "time_to_expiry",
    "rate",
    "is_call",
)

IV_CASES = [
    # (strike, spot, true_vol, t, r, is_call)
    (100.0, 100.0, 0.20, 1.0, 0.05, True),
    (100.0, 100.0, 0.20, 1.0, 0.05, False),
    (120.0, 100.0, 0.30, 0.5, 0.05, True),
    (120.0, 100.0, 0.30, 0.5, 0.05, False),
    (100.0, 50.0, 0.40, 2.0, 0.03, True),  # deep OTM
    (90.0, 100.0, 0.15, 0.1, 0.0, True),  # short dated ITM
    (250.0, 250.0, 1.2, 0.007, 0.0425, True),  # crypto-ish
    (95.0, 100.0, 0.05, 0.75, 0.05, False),  # low vol
    (100.0, 100.0, 1.5, 2.0, -0.02, True),  # high vol, negative rates
]


def _iv_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strike": [c[0] for c in IV_CASES],
            "spot": [c[1] for c in IV_CASES],
            "vol": [c[2] for c in IV_CASES],
            "time_to_expiry": [c[3] for c in IV_CASES],
            "rate": [c[4] for c in IV_CASES],
            "is_call": [c[5] for c in IV_CASES],
        }
    ).with_columns(
        # price the chain with the plugin itself: exact round-trip target
        option_price=black_scholes(
            "strike", "spot", "vol", "time_to_expiry", "rate", "is_call"
        ),
    )


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_implied_vol_round_trip(method):
    df = _iv_frame().with_columns(iv=implied_vol(*IV_ARGS, method=method))
    for case, iv in zip(IV_CASES, df["iv"].to_list()):
        assert math.isclose(iv, case[2], rel_tol=1e-6, abs_tol=1e-8), (case, method, iv)


def test_implied_vol_methods_agree():
    df = _iv_frame().with_columns(
        newton=implied_vol(*IV_ARGS, method="newton"),
        halley=implied_vol(*IV_ARGS, method="halley"),
    )
    for n, h in zip(df["newton"].to_list(), df["halley"].to_list()):
        assert math.isclose(n, h, rel_tol=1e-9, abs_tol=1e-9), (n, h)


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_implied_vol_matches_independent_solver(method):
    # bisection on the erfc-based reference price, independent of the kernel
    def ref_iv(case, target):
        k, s, _v, t, r, is_call = case
        lo, hi = 1e-7, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ref_price(k, s, mid, t, r, is_call) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    df = _iv_frame().with_columns(iv=implied_vol(*IV_ARGS, method=method))
    for case, iv in zip(IV_CASES, df["iv"].to_list()):
        expected = ref_iv(case, ref_price(*case))
        # gap is the A&S CDF error showing up as a small vol shift
        assert math.isclose(iv, expected, abs_tol=1e-5), (case, method, iv, expected)


def test_implied_vol_invalid_method_raises():
    with pytest.raises(pl.exceptions.PolarsError):
        _iv_frame().with_columns(iv=implied_vol(*IV_ARGS, method="bogus"))


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_implied_vol_no_solution_rows_are_nan(method):
    disc = math.exp(-0.05)
    df = pl.DataFrame(
        {
            "option_price": [
                10.450584,  # solvable anchor
                0.0,  # zero price
                -1.0,  # negative price
                max(100.0 - 90.0 * disc, 0.0),  # exactly discounted intrinsic
                max(100.0 - 90.0 * disc, 0.0) - 0.5,  # below intrinsic (arbitrage)
                100.0,  # at the call upper bound (spot)
                250.0,  # above the call upper bound
                1.0,  # t = 0 row: no vol can price it
            ],
            "strike": [100.0, 100.0, 100.0, 90.0, 90.0, 100.0, 100.0, 100.0],
            "spot": [100.0] * 8,
            "time_to_expiry": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "rate": [0.05] * 8,
            "is_call": [True] * 8,
        }
    ).with_columns(iv=implied_vol(*IV_ARGS, method=method))
    ivs = df["iv"].to_list()
    assert not math.isnan(ivs[0])
    assert all(math.isnan(v) for v in ivs[1:])


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_implied_vol_null_propagates(method):
    df = pl.DataFrame(
        {
            "option_price": [10.450584, None],
            "strike": [100.0, 100.0],
            "spot": [100.0, 100.0],
            "time_to_expiry": [1.0, 1.0],
            "rate": [0.05, 0.05],
            "is_call": [True, None],
        }
    ).with_columns(iv=implied_vol(*IV_ARGS, method=method))
    assert df["iv"].to_list()[1] is None


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_backfill_iv_and_greeks_workflow(method):
    # the market-data workflow: prices in, IV + greeks out
    df = (
        _iv_frame()
        .drop("vol")
        .with_columns(iv=implied_vol(*IV_ARGS, method=method))
        .drop("option_price")
        .with_columns(
            greeks=black_scholes_with_greeks(
                "strike", "spot", "iv", "time_to_expiry", "rate", "is_call"
            ),
        )
        .unnest("greeks")
    )

    for case, row in zip(IV_CASES, df.iter_rows(named=True)):
        assert math.isclose(row["iv"], case[2], rel_tol=1e-6), (case, row["iv"])
        expected = ref_greeks(*case)
        for name, exp in zip(GREEK_NAMES, expected):
            assert math.isclose(row[name], exp, rel_tol=1e-5, abs_tol=1e-4), (
                case,
                name,
                row[name],
                exp,
            )


IV_GREEK_FIELDS = ["implied_vol"] + GREEK_NAMES


@pytest.mark.parametrize("method", ["newton", "halley"])
def test_iv_and_greeks_round_trip(method):
    df = (
        _iv_frame()
        .with_columns(market=pl.col("option_price"))
        .with_columns(
            result=iv_and_greeks(*IV_ARGS, method=method),
        )
        .drop("option_price")
        .unnest("result")
    )

    for case, row in zip(IV_CASES, df.iter_rows(named=True)):
        assert math.isclose(row["implied_vol"], case[2], rel_tol=1e-6), (
            case,
            method,
            row["implied_vol"],
        )
        # the model price at the solved vol is the market price
        assert math.isclose(row["model_price"], row["market"], abs_tol=1e-8), case
        expected = ref_greeks(*case)
        for name, exp in zip(GREEK_NAMES, expected):
            assert math.isclose(row[name], exp, rel_tol=1e-5, abs_tol=1e-4), (
                case,
                method,
                name,
                row[name],
                exp,
            )


def test_iv_and_greeks_matches_composition():
    # identical to implied_vol followed by black_scholes_with_greeks
    composed = (
        _iv_frame()
        .with_columns(
            iv=implied_vol(*IV_ARGS),
        )
        .drop("option_price")
        .with_columns(
            greeks=black_scholes_with_greeks(
                "strike", "spot", "iv", "time_to_expiry", "rate", "is_call"
            ),
        )
        .unnest("greeks")
    )
    fused = (
        _iv_frame()
        .with_columns(
            result=iv_and_greeks(*IV_ARGS),
        )
        .drop("option_price")
        .unnest("result")
    )

    assert fused["implied_vol"].equals(composed["iv"])
    for name in GREEK_NAMES:
        assert fused[name].equals(composed[name]), name


def test_iv_and_greeks_no_solution_rows_are_nan():
    disc = math.exp(-0.05)
    df = (
        pl.DataFrame(
            {
                "option_price": [
                    10.450584,  # solvable anchor
                    max(100.0 - 90.0 * disc, 0.0),  # exactly intrinsic
                    250.0,  # above the call upper bound
                    1.0,  # t = 0
                ],
                "strike": [100.0, 90.0, 100.0, 100.0],
                "spot": [100.0] * 4,
                "time_to_expiry": [1.0, 1.0, 1.0, 0.0],
                "rate": [0.05] * 4,
                "is_call": [True] * 4,
            }
        )
        .with_columns(
            result=iv_and_greeks(*IV_ARGS),
        )
        .drop("option_price")
        .unnest("result")
    )

    for i, row in enumerate(df.iter_rows(named=True)):
        for name in IV_GREEK_FIELDS:
            value = row[name]
            if i == 0:
                assert not math.isnan(value), (name, value)
            else:
                assert math.isnan(value), (i, name, value)


def test_iv_and_greeks_null_propagates():
    df = (
        pl.DataFrame(
            {
                "option_price": [10.450584, 10.450584],
                "strike": [100.0, None],
                "spot": [100.0, 100.0],
                "time_to_expiry": [1.0, 1.0],
                "rate": [0.05, 0.05],
                "is_call": [True, True],
            }
        )
        .with_columns(
            result=iv_and_greeks(*IV_ARGS),
        )
        .drop("option_price")
        .unnest("result")
    )
    assert all(df[name].to_list()[1] is None for name in IV_GREEK_FIELDS)
