//! Black-Scholes option price expression.
//!
//! Column inputs, in this order: `strike`, `spot`, `vol`, `time_to_expiry`,
//! `rate`, and `is_call` (a Boolean column: true = call, false = put).
//! `black_scholes` returns the option price as f64;
//! `black_scholes_with_greeks` returns a struct with `price`, `delta`,
//! `gamma`, `vega`, `theta` and `rho`; `implied_vol` takes a market `price`
//! in place of `vol` (inputs: `price, strike, spot, time_to_expiry, rate,
//! is_call`) and solves for the volatility that reprices it; `iv_and_greeks`
//! takes the same inputs and returns the solved volatility plus the Greeks
//! at it, as a struct.
//!
//! Per-row semantics (nulls in any input propagate):
//! - `time_to_expiry <= 0`: the option is worth its (undiscounted) intrinsic
//!   value.
//! - `spot <= 0`, `strike <= 0` or `vol <= 0`: returns 0.0.
//!
//! Greeks conventions: `delta` per unit spot, `gamma` per unit spot squared,
//! `vega` per unit vol (not per vol point), `theta` per year, `rho` per unit
//! rate. Degenerate rows carry `price` as above, `delta` = the exercise
//! indicator when `time_to_expiry <= 0` (0.0 when at-the-money) and 0.0
//! otherwise, and `gamma`/`vega`/`theta`/`rho` = 0.0.
//!
//! The kernels are plain serial loops; both expressions are registered as
//! elementwise, so Polars' streaming engine provides the parallelism by
//! running them concurrently across morsels.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use serde::Deserialize;

const INV_SQRT_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
// FRAC_1_SQRT_2PI is not stable on this toolchain; 1/sqrt(2*pi) in full f64 precision
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Standard normal cumulative distribution function N(x).
///
/// Uses the Abramowitz & Stegun 7.1.26 rational approximation of the error
/// function, accurate to ~1.5e-7 — plenty for option pricing. N(x) needs
/// erf(x/sqrt(2)), so the argument is scaled before applying the approximation
/// (which is only valid for non-negative arguments; odd symmetry handles the
/// negative side).
#[inline]
fn norm_cdf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 coefficients.
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let z = x * INV_SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + P * z.abs());
    let erf = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-z * z).exp();

    0.5 * (1.0 + sign * erf)
}

/// Standard normal probability density function phi(x) — exact, and the
/// building block for the analytical Greeks.
#[inline]
fn norm_pdf(x: f64) -> f64 {
    // exp(-x^2/2) / sqrt(2*pi)
    (-x * x * 0.5).exp() * INV_SQRT_2PI
}

#[inline]
fn compute_d1(
    log_moneyness: f64,
    time_to_expiry: f64,
    vol_sqrt_t: f64,
    rate: f64,
    sigma: f64,
) -> f64 {
    (log_moneyness + (rate + 0.5 * sigma * sigma) * time_to_expiry) / vol_sqrt_t
}

/// Per-row option price.
fn black_scholes_rs(
    strike: f64,
    spot: f64,
    vol: f64,
    time_to_expiry: f64,
    rate: f64,
    is_call: bool,
) -> f64 {
    // At (or past) expiry the option is worth its intrinsic value.
    if time_to_expiry <= 0.0 {
        return if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
    }

    // Degenerate inputs would produce NaN via log(0) / division by zero.
    if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 {
        return 0.0;
    }

    let sqrt_t = time_to_expiry.sqrt();
    let log_moneyness = (spot / strike).ln();
    let d1 = compute_d1(log_moneyness, time_to_expiry, vol * sqrt_t, rate, vol);
    let d2 = d1 - vol * sqrt_t;
    let discount = (-rate * time_to_expiry).exp();

    if is_call {
        spot * norm_cdf(d1) - strike * discount * norm_cdf(d2)
    } else {
        strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1)
    }
}

/// Option price and analytical Greeks for one row.
struct Greeks {
    price: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho: f64,
}

/// Per-row price and analytical Greeks.
fn black_scholes_with_greeks_rs(
    strike: f64,
    spot: f64,
    vol: f64,
    time_to_expiry: f64,
    rate: f64,
    is_call: bool,
) -> Greeks {
    // At (or past) expiry the option is worth its intrinsic value; delta
    // degenerates into the exercise indicator, everything else into zero.
    if time_to_expiry <= 0.0 {
        let price = if is_call {
            (spot - strike).max(0.0)
        } else {
            (strike - spot).max(0.0)
        };
        let delta = if is_call {
            if spot > strike {
                1.0
            } else {
                0.0
            }
        } else if strike > spot {
            -1.0
        } else {
            0.0
        };
        return Greeks {
            price,
            delta,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }

    // Degenerate inputs would produce NaN via log(0) / division by zero.
    if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 {
        return Greeks {
            price: 0.0,
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
    }

    let sqrt_t = time_to_expiry.sqrt();
    let vol_sqrt_t = vol * sqrt_t;
    let log_moneyness = (spot / strike).ln();
    let d1 = compute_d1(log_moneyness, time_to_expiry, vol_sqrt_t, rate, vol);
    let d2 = d1 - vol_sqrt_t;
    let discount = (-rate * time_to_expiry).exp();
    let pdf_d1 = norm_pdf(d1);

    let (price, delta) = if is_call {
        (
            spot * norm_cdf(d1) - strike * discount * norm_cdf(d2),
            norm_cdf(d1),
        )
    } else {
        (
            strike * discount * norm_cdf(-d2) - spot * norm_cdf(-d1),
            norm_cdf(d1) - 1.0,
        )
    };
    // gamma and vega are identical for calls and puts
    let gamma = pdf_d1 / (spot * vol_sqrt_t);
    let vega = spot * pdf_d1 * sqrt_t;
    let theta = if is_call {
        -spot * pdf_d1 * vol / (2.0 * sqrt_t) - rate * strike * discount * norm_cdf(d2)
    } else {
        -spot * pdf_d1 * vol / (2.0 * sqrt_t) + rate * strike * discount * norm_cdf(-d2)
    };
    let rho = if is_call {
        strike * time_to_expiry * discount * norm_cdf(d2)
    } else {
        -strike * time_to_expiry * discount * norm_cdf(-d2)
    };

    Greeks {
        price,
        delta,
        gamma,
        vega,
        theta,
        rho,
    }
}

/// Root-finding method for the implied volatility solver.
#[derive(Clone, Copy, Deserialize)]
enum IvMethod {
    #[serde(rename = "newton")]
    Newton,
    #[serde(rename = "halley")]
    Halley,
}

/// Solve the implied volatility that reprices `price`.
///
/// Bracketed root finding on the same A&S-based formula the pricing kernels
/// use: each step is a Newton jump off vega (quadratic convergence) or, when
/// `method` is Halley, a cubic jump off vega and volga; both fall back to
/// bisection when vega is negligible or the jump leaves the bracket, and the
/// bracket tightens from the model-versus-market sign every iteration.
/// Returns NaN when no solution exists: non-positive inputs, or a price
/// outside the no-arbitrage band (discounted intrinsic, spot for a call /
/// discounted strike for a put), which also covers IVs beyond the
/// [1e-7, 10.0] search bracket.
fn implied_vol_rs(
    price: f64,
    strike: f64,
    spot: f64,
    time_to_expiry: f64,
    rate: f64,
    is_call: bool,
    method: IvMethod,
) -> f64 {
    // NaN inputs fail these comparisons too
    if !(time_to_expiry > 0.0 && spot > 0.0 && strike > 0.0 && price > 0.0) {
        return f64::NAN;
    }

    let disc = (-rate * time_to_expiry).exp();
    let lower = if is_call {
        (spot - strike * disc).max(0.0)
    } else {
        (strike * disc - spot).max(0.0)
    };
    let upper = if is_call { spot } else { strike * disc };
    if price <= lower || price >= upper {
        return f64::NAN;
    }

    const SIGMA_LO: f64 = 1e-7;
    const SIGMA_HI: f64 = 10.0;
    const SIGMA_TOL: f64 = 1e-12;
    const MAX_ITER: usize = 100;

    let price_tol = 1e-12 * (spot + strike);
    let mut lo = SIGMA_LO;
    let mut hi = SIGMA_HI;
    // Brenner-Subrahmanyam at-the-money guess, clamped into a sane band
    let mut sigma =
        ((price / spot) * (std::f64::consts::TAU / time_to_expiry).sqrt()).clamp(0.05, 3.0);

    let sqrt_t = time_to_expiry.sqrt();
    let log_moneyness = (spot / strike).ln();
    for _ in 0..MAX_ITER {
        let model = black_scholes_rs(strike, spot, sigma, time_to_expiry, rate, is_call);
        let diff = model - price;
        if diff.abs() <= price_tol {
            return sigma;
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        if hi - lo <= SIGMA_TOL {
            // the bracket collapsed without repricing the market price:
            // no volatility in [SIGMA_LO, SIGMA_HI] solves this row
            return f64::NAN;
        }
        let vol_sqrt_t = sigma * sqrt_t;
        let d1 = compute_d1(log_moneyness, time_to_expiry, vol_sqrt_t, rate, sigma);
        let vega = spot * norm_pdf(d1) * sqrt_t;
        let candidate = if vega > 1e-10 * spot {
            let newton = sigma - diff / vega;
            match method {
                IvMethod::Newton => newton,
                IvMethod::Halley => {
                    // Halley's cubic step: the Newton jump divided by
                    // 1 - f*f''/(2*f'^2), with volga = vega*d1*d2/sigma as f''
                    let volga = vega * d1 * (d1 - vol_sqrt_t) / sigma; // volga or vomma, the second derivative of price w.r.t. vol
                    let denom = 1.0 - diff * volga / (2.0 * vega * vega);
                    if denom > 1e-8 {
                        sigma - (diff / vega) / denom
                    } else {
                        newton
                    }
                },
            }
        } else {
            0.5 * (lo + hi)
        };
        sigma = if candidate > lo && candidate < hi {
            candidate
        } else {
            0.5 * (lo + hi)
        };
    }
    f64::NAN
}

/// Broadcast a series to `len` (length-1 inputs repeat).
fn broadcast_to(s: &Series, len: usize) -> PolarsResult<Series> {
    if s.len() == len {
        Ok(s.clone())
    } else if s.len() == 1 {
        Ok(s.new_from_index(0, len))
    } else {
        polars_bail!(ShapeMismatch: format!(
            "cannot broadcast black-scholes input of length {} to length {}", s.len(), len
        ));
    }
}

/// Common input plumbing: the five numeric inputs cast to f64, plus the
/// Boolean `is_call` column, all broadcast to a common length.
pub(crate) fn bs_inputs(inputs: &[Series]) -> PolarsResult<(Vec<Series>, BooleanChunked)> {
    polars_ensure!(inputs.len() == 6, InvalidOperation: "black-scholes expects 6 inputs");
    let len = inputs.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut numeric = Vec::with_capacity(5);
    for s in &inputs[..5] {
        let s = s.cast(&DataType::Float64)?;
        numeric.push(broadcast_to(&s, len)?);
    }
    let is_call = broadcast_to(&inputs[5], len)?;
    Ok((numeric, is_call.bool()?.clone()))
}

#[polars_expr(output_type=Float64)]
fn black_scholes(inputs: &[Series]) -> PolarsResult<Series> {
    let (args, is_call) = bs_inputs(inputs)?;
    let (k, s, v, t, r) = (
        args[0].f64()?,
        args[1].f64()?,
        args[2].f64()?,
        args[3].f64()?,
        args[4].f64()?,
    );
    let out: Float64Chunked = k
        .iter()
        .zip(s.iter())
        .zip(v.iter())
        .zip(t.iter())
        .zip(r.iter())
        .zip(is_call.iter())
        .map(|(((((k, s), v), t), r), ic)| match (k, s, v, t, r, ic) {
            (Some(k), Some(s), Some(v), Some(t), Some(r), Some(ic)) => {
                Some(black_scholes_rs(k, s, v, t, r, ic))
            },
            _ => None,
        })
        .collect_trusted();
    Ok(out.into_series())
}

#[derive(Deserialize)]
struct ImpliedVolKwargs {
    method: IvMethod,
}

#[polars_expr(output_type=Float64)]
fn implied_vol(inputs: &[Series], kwargs: ImpliedVolKwargs) -> PolarsResult<Series> {
    let (args, is_call) = bs_inputs(inputs)?;
    let (p, k, s, t, r) = (
        args[0].f64()?,
        args[1].f64()?,
        args[2].f64()?,
        args[3].f64()?,
        args[4].f64()?,
    );
    let method = kwargs.method;
    let out: Float64Chunked = p
        .iter()
        .zip(k.iter())
        .zip(s.iter())
        .zip(t.iter())
        .zip(r.iter())
        .zip(is_call.iter())
        .map(|(((((p, k), s), t), r), ic)| match (p, k, s, t, r, ic) {
            (Some(p), Some(k), Some(s), Some(t), Some(r), Some(ic)) => {
                Some(implied_vol_rs(p, k, s, t, r, ic, method))
            },
            _ => None,
        })
        .collect_trusted();
    Ok(out.into_series())
}

fn black_scholes_with_greeks_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "black_scholes_with_greeks".into(),
        DataType::Struct(vec![
            Field::new("model_price".into(), DataType::Float64),
            Field::new("delta".into(), DataType::Float64),
            Field::new("gamma".into(), DataType::Float64),
            Field::new("vega".into(), DataType::Float64),
            Field::new("theta".into(), DataType::Float64),
            Field::new("rho".into(), DataType::Float64),
        ]),
    ))
}

/// Price and analytical Greeks in a single pass, returned as a struct.
#[polars_expr(output_type_func=black_scholes_with_greeks_output)]
fn black_scholes_with_greeks(inputs: &[Series]) -> PolarsResult<Series> {
    let (args, is_call) = bs_inputs(inputs)?;
    let (k, s, v, t, r) = (
        args[0].f64()?,
        args[1].f64()?,
        args[2].f64()?,
        args[3].f64()?,
        args[4].f64()?,
    );

    let mut prices: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut deltas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut gammas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut vegas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut thetas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut rhos: Vec<Option<f64>> = Vec::with_capacity(is_call.len());

    for (((((k, s), v), t), r), ic) in k
        .iter()
        .zip(s.iter())
        .zip(v.iter())
        .zip(t.iter())
        .zip(r.iter())
        .zip(is_call.iter())
    {
        match (k, s, v, t, r, ic) {
            (Some(k), Some(s), Some(v), Some(t), Some(r), Some(ic)) => {
                let g = black_scholes_with_greeks_rs(k, s, v, t, r, ic);
                prices.push(Some(g.price));
                deltas.push(Some(g.delta));
                gammas.push(Some(g.gamma));
                vegas.push(Some(g.vega));
                thetas.push(Some(g.theta));
                rhos.push(Some(g.rho));
            },
            _ => {
                prices.push(None);
                deltas.push(None);
                gammas.push(None);
                vegas.push(None);
                thetas.push(None);
                rhos.push(None);
            },
        }
    }

    let fields = [
        Float64Chunked::from_slice_options("model_price".into(), &prices).into_series(),
        Float64Chunked::from_slice_options("delta".into(), &deltas).into_series(),
        Float64Chunked::from_slice_options("gamma".into(), &gammas).into_series(),
        Float64Chunked::from_slice_options("vega".into(), &vegas).into_series(),
        Float64Chunked::from_slice_options("theta".into(), &thetas).into_series(),
        Float64Chunked::from_slice_options("rho".into(), &rhos).into_series(),
    ];
    let ca = StructChunked::from_series(
        "black_scholes_with_greeks".into(),
        is_call.len(),
        fields.iter(),
    )?;
    Ok(ca.into_series())
}

fn iv_and_greeks_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "iv_and_greeks".into(),
        DataType::Struct(vec![
            Field::new("implied_vol".into(), DataType::Float64),
            Field::new("model_price".into(), DataType::Float64),
            Field::new("delta".into(), DataType::Float64),
            Field::new("gamma".into(), DataType::Float64),
            Field::new("vega".into(), DataType::Float64),
            Field::new("theta".into(), DataType::Float64),
            Field::new("rho".into(), DataType::Float64),
        ]),
    ))
}

/// Market price in: implied volatility and analytical Greeks out, in one
/// expression. Solves the vol with `implied_vol_rs`, then evaluates the
/// Greeks kernel at it. Rows where no vol reprices the quote are NaN
/// throughout; null inputs stay null.
#[polars_expr(output_type_func=iv_and_greeks_output)]
fn iv_and_greeks(inputs: &[Series], kwargs: ImpliedVolKwargs) -> PolarsResult<Series> {
    let (args, is_call) = bs_inputs(inputs)?;
    let (p, k, s, t, r) = (
        args[0].f64()?,
        args[1].f64()?,
        args[2].f64()?,
        args[3].f64()?,
        args[4].f64()?,
    );
    let method = kwargs.method;

    let mut ivs: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut prices: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut deltas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut gammas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut vegas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut thetas: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut rhos: Vec<Option<f64>> = Vec::with_capacity(is_call.len());

    for (((((p, k), s), t), r), ic) in p
        .iter()
        .zip(k.iter())
        .zip(s.iter())
        .zip(t.iter())
        .zip(r.iter())
        .zip(is_call.iter())
    {
        match (p, k, s, t, r, ic) {
            (Some(p), Some(k), Some(s), Some(t), Some(r), Some(ic)) => {
                let sigma = implied_vol_rs(p, k, s, t, r, ic, method);
                if sigma.is_nan() {
                    // no vol reprices the quote: no numbers for this row
                    let nan = Some(f64::NAN);
                    ivs.push(nan);
                    prices.push(nan);
                    deltas.push(nan);
                    gammas.push(nan);
                    vegas.push(nan);
                    thetas.push(nan);
                    rhos.push(nan);
                } else {
                    let g = black_scholes_with_greeks_rs(k, s, sigma, t, r, ic);
                    ivs.push(Some(sigma));
                    prices.push(Some(g.price));
                    deltas.push(Some(g.delta));
                    gammas.push(Some(g.gamma));
                    vegas.push(Some(g.vega));
                    thetas.push(Some(g.theta));
                    rhos.push(Some(g.rho));
                }
            },
            _ => {
                ivs.push(None);
                prices.push(None);
                deltas.push(None);
                gammas.push(None);
                vegas.push(None);
                thetas.push(None);
                rhos.push(None);
            },
        }
    }

    let fields = [
        Float64Chunked::from_slice_options("implied_vol".into(), &ivs).into_series(),
        Float64Chunked::from_slice_options("model_price".into(), &prices).into_series(),
        Float64Chunked::from_slice_options("delta".into(), &deltas).into_series(),
        Float64Chunked::from_slice_options("gamma".into(), &gammas).into_series(),
        Float64Chunked::from_slice_options("vega".into(), &vegas).into_series(),
        Float64Chunked::from_slice_options("theta".into(), &thetas).into_series(),
        Float64Chunked::from_slice_options("rho".into(), &rhos).into_series(),
    ];
    let ca = StructChunked::from_series("iv_and_greeks".into(), is_call.len(), fields.iter())?;
    Ok(ca.into_series())
}
