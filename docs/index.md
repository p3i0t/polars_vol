# polars-vol

Rust-powered [Polars](https://pola.rs) expressions for option pricing and
volatility work — Black-Scholes prices, analytical Greeks and implied
volatility, all elementwise and columnar. The
[README](https://github.com/p3i0t/polars_vol) has the full API tour with
code examples.

## Getting Started

- [Installation](install.md) — pip, uv, and building from source.
- [Quick Start](https://github.com/p3i0t/polars_vol#quick-start) — pricing
  and IV backfill in a few lines (in the README).

## User Guide

- [Black-Scholes speed comparison](black_scholes_speed.md) — the plugin vs
  vectorized numpy and scipy, and why the streaming engine matters.
- [Newton vs Halley for implied volatility](newton_vs_halley.md) — the math
  behind the two root-finding methods, with runnable code.
