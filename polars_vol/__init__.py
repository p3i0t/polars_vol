from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_vol._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_vol.typing import IntoExprColumn

LIB = Path(__file__).parent


def black_scholes(
    strike: IntoExprColumn,
    spot: IntoExprColumn,
    vol: IntoExprColumn,
    time_to_expiry: IntoExprColumn,
    rate: IntoExprColumn,
    is_call: IntoExprColumn,
) -> pl.Expr:
    """Price a European option with the Black-Scholes formula.

    Parameters
    ----------
    strike : strike price K
    spot : underlying price S
    vol : annualized volatility
    time_to_expiry : time to expiry T, in years
    rate : continuously compounded risk-free rate r
    is_call : Boolean column; True for a call, False for a put. A length-1
        input such as ``pl.lit(True)`` broadcasts against the other columns.

    Rows with a null in any input are null. `time_to_expiry <= 0` gives the
    undiscounted intrinsic value; `spot <= 0`, `strike <= 0` or `vol <= 0`
    give 0.0.
    """
    return register_plugin_function(
        args=[strike, spot, vol, time_to_expiry, rate, is_call],
        plugin_path=LIB,
        function_name="black_scholes",
        is_elementwise=True,
    )


def black_scholes_with_greeks(
    strike: IntoExprColumn,
    spot: IntoExprColumn,
    vol: IntoExprColumn,
    time_to_expiry: IntoExprColumn,
    rate: IntoExprColumn,
    is_call: IntoExprColumn,
) -> pl.Expr:
    """Price a European option and compute its analytical Greeks.

    Takes the same arguments as :func:`black_scholes` and returns a struct
    with fields ``model_price``, ``delta``, ``gamma``, ``vega``, ``theta``
    and ``rho``; use ``.unnest("black_scholes_with_greeks")`` to expand them
    into columns.

    Conventions (canonical analytic units): ``delta`` per unit spot,
    ``gamma`` per unit spot squared, ``vega`` per unit annualized vol,
    ``theta`` per year, ``rho`` per unit rate. Only theta, vega and rho
    carry a unit worth converting to market quoting conventions:

    >>> df.with_columns(
    ...     theta_per_day=pl.col("theta") / 365,  # or / 252 for trading days
    ...     vega_per_vp=pl.col("vega") / 100,  # per vol point
    ...     rho_per_pct=pl.col("rho") / 100,  # per 1% rate
    ... ...)

    ``delta`` and ``gamma`` need no conversion. The daily theta is the
    instantaneous decay at today's ``time_to_expiry``.

    Rows with a null in any input are null in all fields. On degenerate rows
    the price follows :func:`black_scholes`; ``delta`` is the exercise
    indicator when ``time_to_expiry <= 0`` (0.0 when at-the-money) and 0.0
    otherwise, and the remaining Greeks are 0.0.
    """
    return register_plugin_function(
        args=[strike, spot, vol, time_to_expiry, rate, is_call],
        plugin_path=LIB,
        function_name="black_scholes_with_greeks",
        is_elementwise=True,
    )


def implied_vol(
    option_price: IntoExprColumn,
    strike: IntoExprColumn,
    spot: IntoExprColumn,
    time_to_expiry: IntoExprColumn,
    rate: IntoExprColumn,
    is_call: IntoExprColumn,
    *,
    method: Literal["newton", "halley"] = "newton",
) -> pl.Expr:
    """Solve the Black-Scholes implied volatility from an option's market
    price.

    Parameters
    ----------
    option_price : observed option price (e.g. the mid quote)
    strike, spot, time_to_expiry, rate, is_call : as in :func:`black_scholes`
    method : ``"newton"`` (default) or ``"halley"`` — the per-iteration step
        for the root finder. Halley converges cubically using the volga as
        the second derivative and typically needs one iteration fewer; both
        share the same bisection safeguard and produce identical results.

    Uses bracketed root finding on the same formula the pricing kernels use
    (typically ~5 Newton / ~3 Halley iterations). The search bracket is
    ``[1e-7, 10.0]``.

    Rows are NaN when no volatility can reprice the price: non-positive
    inputs, a price outside the no-arbitrage band (below the discounted
    intrinsic value, or at/above spot for a call / the discounted strike for
    a put), or an implied volatility beyond the bracket. Null inputs stay
    null. IV accuracy is limited by the A&S CDF error, in practice ~1e-7 in
    volatility units.

    Combine with the greeks to backfill a chain::

        df.with_columns(iv=implied_vol("mid", *ARGS)).with_columns(
            greeks=black_scholes_with_greeks(
                "strike", "spot", "iv", "time_to_expiry", "rate", "is_call"
            )
        ).unnest("greeks")
    """
    return register_plugin_function(
        args=[option_price, strike, spot, time_to_expiry, rate, is_call],
        plugin_path=LIB,
        function_name="implied_vol",
        is_elementwise=True,
        kwargs={"method": method},
    )


def iv_and_greeks(
    option_price: IntoExprColumn,
    strike: IntoExprColumn,
    spot: IntoExprColumn,
    time_to_expiry: IntoExprColumn,
    rate: IntoExprColumn,
    is_call: IntoExprColumn,
    *,
    method: Literal["newton", "halley"] = "newton",
) -> pl.Expr:
    """Backfill implied volatility and Greeks from market prices in one
    expression.

    Takes the same arguments as :func:`implied_vol` (including the `method`
    keyword) and returns a struct with fields ``implied_vol``,
    ``model_price``, ``delta``, ``gamma``, ``vega``, ``theta`` and ``rho``;
    use ``.unnest("iv_and_greeks")`` to expand them into columns. The Greeks
    are the analytical Black-Scholes Greeks evaluated at the solved
    volatility; ``model_price`` is the model price at that vol (i.e. the
    market price, up to solver tolerance). Greek units and the conversion
    to market quoting conventions (daily theta, vega per vol point, ...)
    are documented on :func:`black_scholes_with_greeks`.

    Rows with a null in any input are null in all fields. Rows where no
    volatility reprices the quote (see :func:`implied_vol`) are NaN in all
    fields.

    >>> df.with_columns(result=iv_and_greeks("mid", *ARGS)).unnest("result")
    """
    return register_plugin_function(
        args=[option_price, strike, spot, time_to_expiry, rate, is_call],
        plugin_path=LIB,
        function_name="iv_and_greeks",
        is_elementwise=True,
        kwargs={"method": method},
    )
