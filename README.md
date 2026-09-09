# polars-vol

A polars plugin implemented in rust for volatility trading.

## Installation

From PyPI, with pip or uv:

```bash
pip install polars-vol
uv add polars-vol            # in a uv-managed project
uv pip install polars-vol    # with the uv pip interface
```

From source (requires a Rust toolchain and
[maturin](https://www.maturin.rs)):

```bash
git clone https://github.com/p3i0t/polars_vol && cd polars_vol
pip install maturin          # or: uv pip install maturin
maturin develop --release
```

Wheels are abi3 (CPython ≥ 3.9) for macOS, Linux and Windows; polars ≥ 1.3
is installed automatically as a dependency.

## Features

The following list are implemented as polars.Expr: 

- `black_scholes(strike, spot, vol, time_to_expiry, rate, is_call)` —
  European call/put price
- `black_scholes_with_greeks(strike, spot, vol, ...)` — price and analytical
  Greeks (`model_price, delta, gamma, vega, theta, rho`) in a single pass
- `implied_vol(option_price, ...)` — implied volatility from market prices,
  via bracketed Newton or Halley (`method="halley"`) iteration
- `iv_and_greeks(option_price, ...)` — the solved implied volatility plus
  all Greeks in one expression

Thanks to the polars *collect(engine="streaming")*, we can easily get **>5x** speedup than vectorized numpy. 

## Quick Start


```python
import polars as pl
import polars_vol as pv

df = pl.DataFrame({
  "strike": [100, 105, 110],
  "spot": [100, 100, 100],
  "time_to_expiry": [0.5, 0.5, 0.5]
}).with_columns(rate=pl.lit(0.05), vol=pl.lit(0.2), is_call=pl.lit(True))

df = df.with_columns(
    option_price=pv.black_scholes(
        "strike", "spot", "vol", "time_to_expiry", "rate", "is_call"
    )
)
print(df)

# backfill iv from option price
iv = df.select(pv.implied_vol(
    "option_price", "strike", "spot", "time_to_expiry", "rate", "is_call"
).alias('iv'))
print(iv)

# backfill iv and greeks from option price
df.select(result=pv.iv_and_greeks(
    "option_price", "strike", "spot", "time_to_expiry", "rate", "is_call", method="newton"  # or "halley"
)).unnest("result")
```

## Examples

### SPX backfill

You may also want to see a real example on SPX [`examples/spx_backfill.py`](examples/spx_backfill.py), where it fetches the SPX chain
from Yahoo, solves the implied volatility of every option from its market
mid with `implied_vol`, and checks the result against Yahoo's own
`impliedVolatility`: correlation ≈ 0.997, median gap ≈ 0.003 vol points. (needs `pip install yfinance`).

## Speed — vs vectorized numpy

Same data, same math — a hand-vectorized numpy/scipy implementation vs the
plugin on Polars' streaming engine:

```python
import timeit

import numpy as np
import polars as pl
from scipy.special import ndtr

from polars_vol import black_scholes

vol, t, r = 0.25, 1.0, 0.05
sq = vol * t**0.5

n = 20_000_000
df = pl.select(
    strike=90 + (pl.int_range(n) % 21).cast(pl.Float64),
    spot=(100 + pl.int_range(n) % 40).cast(pl.Float64),
    is_call=(pl.int_range(n) % 2 == 0),
)
k, s, ic = df["strike"].to_numpy(), df["spot"].to_numpy(), df["is_call"].to_numpy()
lf = df.lazy().with_columns(
    price=black_scholes("strike", "spot", pl.lit(vol), pl.lit(t), pl.lit(r), "is_call")
)


def numpy_price():
    d1 = (np.log(s / k) + (r + vol * vol / 2) * t) / sq
    d2 = d1 - sq
    return np.where(
        ic,
        s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2),
        k * np.exp(-r * t) * ndtr(-d2) - s * ndtr(-d1),
    )


def plugin_price():
    return lf.collect(engine="streaming").get_column("price").to_numpy()


t_np = min(timeit.repeat(numpy_price, number=1, repeat=3))
t_pl = min(timeit.repeat(plugin_price, number=1, repeat=3))
diff = (plugin_price() - numpy_price()).__abs__().max()
print(
    f"numpy {t_np:.2f}s | polars_vol {t_pl:.2f}s | {t_np / t_pl:.0f}x | max diff {diff:.1e}"
)
# -> numpy 0.46s | polars_vol 0.05s | 8x | max diff 1.5e-05  (the A&S 7.1.26 CDF error)
```

check [`examples/speed_vs_numpy.py`](examples/speed_vs_numpy.py) for the full code.

MIT license.
