# polars-vol

Rust-powered [Polars](https://pola.rs) expressions for option pricing and
volatility work. See the [README](https://github.com/p3i0t/polars_vol) for
installation and the full API; the pages below go deeper into two topics:

- [Installation](install.md) — pip, uv, and building from source.
- [Black-Scholes speed comparison](black_scholes_speed.md) — the plugin vs
  vectorized numpy and scipy, and why the streaming engine matters.
- [Newton vs Halley for implied volatility](newton_vs_halley.md) — the math
  behind the two root-finding methods, with runnable code.
