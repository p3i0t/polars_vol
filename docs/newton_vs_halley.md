# Newton vs Halley for implied volatility

Implied volatility is a root-finding problem in disguise: the Black-Scholes
price is strictly increasing in σ, so the vol that matches a market quote
solves `f(σ) = BS(σ) − market_price = 0`.

**Newton** follows the tangent line and converges quadratically (correct
digits double per step); **Halley** uses the second derivative as well and
converges cubically (digits triple):

```
Newton:  σ ← σ − f/f′                    f′ = vega  = S·φ(d1)·√T
Halley:  σ ← σ − (f/f′)/(1 − c)          c  = f·f″/(2·f′²),  f″ = volga = vega·d1·d2/σ
```

All derivatives are closed-form, so Halley costs a few extra flops per
iteration, nothing more. On a 7-day at-the-money option (true σ = 1.2):

| iteration | Newton \|f\| | Halley \|f\| |
|---|---:|---:|
| 1 | 3.1e-02 | 3.1e-02 |
| 2 | 1.2e-07 | 1.3e-10 |
| 3 | 1.4e-14 | 0.0 |

The core of each solver, minus safeguards:

```python
def iv_newton(price, s, k, t, r):
    sigma = (price / s) * (2.0 * math.pi / t) ** 0.5  # Brenner-Subrahmanyam guess
    while abs((f := bs(s, k, t, r, sigma) - price)) > 1e-12:
        sigma -= f / vega(s, k, t, r, sigma)
    return sigma

def iv_halley(price, s, k, t, r):
    sigma = (price / s) * (2.0 * math.pi / t) ** 0.5
    while abs((f := bs(s, k, t, r, sigma) - price)) > 1e-12:
        v = vega(s, k, t, r, sigma)
        sigma -= (f / v) / (1 - f * volga(s, k, t, r, sigma) / (2 * v * v))
    return sigma
```

`polars_vol.implied_vol(..., method="newton" | "halley")` wraps both steps
in a `[1e-7, 10]` bracket with a bisection fallback, making them equally
robust. In batch backfills the wall-clock difference is nil — each
iteration is dominated by the two CDF evaluations both methods share, and
Halley only saves about one of ~5 iterations. Halley earns its keep when
each function evaluation is expensive (PDE pricing, calibration inner
loops). Either way, deep-ITM/OTM quotes barely pin σ (vega → 0): the
recovered IV is only determined to about price-precision / vega.
