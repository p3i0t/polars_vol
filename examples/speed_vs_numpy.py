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
    return lf.collect(engine="streaming")["price"].to_numpy()


t_np = min(timeit.repeat(numpy_price, number=1, repeat=3))
t_pl = min(timeit.repeat(plugin_price, number=1, repeat=3))
diff = (plugin_price() - numpy_price()).__abs__().max()
print(
    f"numpy {t_np:.2f}s | polars_vol {t_pl:.2f}s | {t_np / t_pl:.0f}x | max diff {diff:.1e}"
)
# -> numpy 0.46s | polars_vol 0.05s | 8x | max diff 1.5e-05  (the A&S 7.1.26 CDF error)
