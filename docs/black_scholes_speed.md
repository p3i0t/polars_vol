# Black-Scholes speed comparison

20M option prices on a 16-core laptop, best of 3 after warm-up, identical
data and identical A&S 7.1.26 CDF in every implementation:

| implementation | time | throughput |
|---|---:|---:|
| numpy, A&S 7.1.26 CDF | 0.36 s | 56 M rows/s |
| numpy, `scipy.special.ndtr` | 0.32 s | 63 M rows/s |
| plugin, in-memory engine | 0.42 s | 47 M rows/s |
| **plugin, streaming engine** | **0.04 s** | **452 M rows/s** |

With Greeks (six outputs per row): numpy 0.48 s vs plugin streaming 0.07 s
(283 M rows/s). Implied volatility: 20M quotes solved in ~0.4 s (~48 M
rows/s). Outputs match the numpy A&S reference to 1.4e-14; against scipy's
machine-precision CDF the maximum price difference is 1.5e-5 — the
documented A&S error at spot ~100.

Why the streaming engine wins by ~8x:

- the expressions are registered elementwise, so the streaming engine
  evaluates morsels concurrently — ~16 plain serial Rust loops in flight;
- the kernel makes one fused pass with no temporaries, where numpy
  allocates and streams through ~20 intermediate arrays;
- the A&S rational CDF is one `exp` plus a few flops.

Streaming requires lazy:
`df.lazy().with_columns(...).collect(engine="streaming")`.

Reproduce with `PYTHONPATH=. python examples/speed_vs_numpy.py`.
