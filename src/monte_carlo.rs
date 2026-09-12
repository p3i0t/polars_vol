//! Monte Carlo option price expression.
//!
//! Column inputs, in this order: `strike`, `spot`, `vol`, `time_to_expiry`,
//! `rate`, and `is_call` (a Boolean column: true = call, false = put) — the
//! same convention as `black_scholes`. `monte_carlo` prices the European
//! option by simulating `n_paths` terminal stock prices per row under the
//! lognormal geometric-Brownian-motion model and averaging the discounted
//! payoffs. It returns a struct with `price` and `error` fields, where
//! `error` is the standard error of the Monte Carlo mean for that row.
//!
//! The `n_paths` standard-normal draws are generated once from a seeded RNG
//! (see `seed`) and shared by every row (common random numbers), so the
//! per-row computation is a pure function of that row's inputs plus one
//! fixed draw vector. Results are deterministic for a given `seed` and
//! identical regardless of how Polars partitions the input across morsels,
//! which makes the expression safely elementwise.
//!
//! Per-row semantics (nulls in any input propagate):
//! - `time_to_expiry <= 0`: the option is worth its (undiscounted) intrinsic
//!   value.
//! - `spot <= 0`, `strike <= 0` or `vol <= 0`: returns 0.0.
//!
//! Accuracy scales as `1/sqrt(n_paths)`; unlike the closed-form kernels this
//! module does not use `norm_cdf`/`norm_pdf`, only the sampler.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

use crate::black_scholes::bs_inputs;

/// Keyword arguments: number of simulated paths per row and the RNG seed.
#[derive(Deserialize)]
struct MonteCarloKwargs {
    #[serde(default = "default_n_paths")]
    n_paths: usize,
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_n_paths() -> usize {
    100_000
}

fn default_seed() -> u64 {
    42
}

/// Price and Monte Carlo standard error for one row.
struct McEstimate {
    price: f64,
    error: f64,
}

/// Average discounted payoff over the `z` standard-normal draws for one row,
/// along with the standard error of that mean (Welford's online algorithm).
#[inline]
fn mc_option_price_rs(
    spot: f64,
    strike: f64,
    rate: f64,
    time_to_expiry: f64,
    sigma: f64,
    is_call: bool,
    z: &[f64],
) -> McEstimate {
    let drift = (rate - 0.5 * sigma * sigma) * time_to_expiry;
    let vol_sqrt_t = sigma * time_to_expiry.sqrt();
    let discount = (-rate * time_to_expiry).exp();

    let mut running_mean = 0.0;
    let mut running_m2 = 0.0;
    let mut count = 0.0;
    for &z in z {
        // Terminal stock price under GBM: S_T = S0 * exp(drift + sigma * sqrt(T) * z)
        let st = spot * (drift + vol_sqrt_t * z).exp();
        let payoff = if is_call {
            (st - strike).max(0.0)
        } else {
            (strike - st).max(0.0)
        };
        let px = payoff * discount;
        count += 1.0;
        let delta = px - running_mean;
        running_mean += delta / count;
        running_m2 += delta * (px - running_mean);
    }
    let var = if count > 1.0 {
        running_m2 / (count - 1.0)
    } else {
        0.0
    };
    McEstimate {
        price: running_mean,
        error: (var / count).sqrt(),
    }
}

fn monte_carlo_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        "monte_carlo".into(),
        DataType::Struct(vec![
            Field::new("price".into(), DataType::Float64),
            Field::new("error".into(), DataType::Float64),
        ]),
    ))
}

#[polars_expr(output_type_func=monte_carlo_output)]
fn monte_carlo(inputs: &[Series], kwargs: MonteCarloKwargs) -> PolarsResult<Series> {
    let (args, is_call) = bs_inputs(inputs)?;
    let (k, s, v, t, r) = (
        args[0].f64()?,
        args[1].f64()?,
        args[2].f64()?,
        args[3].f64()?,
        args[4].f64()?,
    );
    polars_ensure!(kwargs.n_paths > 0, InvalidOperation: "n_paths must be positive");
    let n_paths = kwargs.n_paths;

    // One shared vector of draws for every row: deterministic per seed and
    // independent of any partitioning of the input column.
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(kwargs.seed);
    let draws: Vec<f64> = (0..n_paths).map(|_| normal.sample(&mut rng)).collect();

    let mut prices: Vec<Option<f64>> = Vec::with_capacity(is_call.len());
    let mut errors: Vec<Option<f64>> = Vec::with_capacity(is_call.len());

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
                // Degenerate rows follow the black_scholes convention and
                // carry zero Monte Carlo error (they are deterministic).
                if t <= 0.0 {
                    prices.push(Some(if ic {
                        (s - k).max(0.0)
                    } else {
                        (k - s).max(0.0)
                    }));
                    errors.push(Some(0.0));
                } else if s <= 0.0 || k <= 0.0 || v <= 0.0 {
                    prices.push(Some(0.0));
                    errors.push(Some(0.0));
                } else {
                    let est = mc_option_price_rs(s, k, r, t, v, ic, &draws);
                    prices.push(Some(est.price));
                    errors.push(Some(est.error));
                }
            },
            _ => {
                prices.push(None);
                errors.push(None);
            },
        }
    }

    let fields = [
        Float64Chunked::from_slice_options("price".into(), &prices).into_series(),
        Float64Chunked::from_slice_options("error".into(), &errors).into_series(),
    ];
    let ca = StructChunked::from_series("monte_carlo".into(), is_call.len(), fields.iter())?;
    Ok(ca.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Welford accumulator must agree with a naive two-pass reference.
    #[test]
    fn welford_matches_two_pass_reference() {
        let z = vec![0.5, -0.5, 1.0, -1.0, 0.0];
        let (s0, k, r, t, sigma) = (100.0, 100.0, 0.0, 1.0, 0.2);

        let est = mc_option_price_rs(s0, k, r, t, sigma, true, &z);

        let drift = -0.5 * sigma * sigma;
        let vol_sqrt_t = sigma * t.sqrt();
        let discount = (-r * t).exp();
        let xs: Vec<f64> = z
            .iter()
            .map(|&zi| {
                let st = s0 * (drift + vol_sqrt_t * zi).exp();
                (st - k).max(0.0) * discount
            })
            .collect();
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();

        assert!((est.price - mean).abs() < 1e-12);
        assert!((est.error - se).abs() < 1e-12);
    }

    /// With zero vol the terminal price is deterministic, so the price is the
    /// discounted intrinsic value and the error is exactly zero.
    #[test]
    fn zero_vol_is_discounted_intrinsic_with_zero_error() {
        let (s0, k, r, t) = (100.0, 95.0, 0.05, 0.5);
        let z = vec![0.0, 0.0, 0.0];

        let call = mc_option_price_rs(s0, k, r, t, 0.0, true, &z);
        let expected_call = (s0 - k * (-r * t).exp()).max(0.0);
        assert!((call.price - expected_call).abs() < 1e-12);
        assert_eq!(call.error, 0.0);

        let put = mc_option_price_rs(s0, k, r, t, 0.0, false, &z);
        let expected_put = (k * (-r * t).exp() - s0).max(0.0);
        assert!((put.price - expected_put).abs() < 1e-12);
        assert_eq!(put.error, 0.0);
    }

    /// Put-call parity must hold exactly in the deterministic (zero vol) limit.
    #[test]
    fn put_call_parity_at_zero_vol() {
        let (s0, k, r, t) = (100.0, 95.0, 0.05, 0.5);
        let z = vec![1.0, -1.0, 0.5]; // arbitrary: sigma = 0 ignores the draws

        let call = mc_option_price_rs(s0, k, r, t, 0.0, true, &z);
        let put = mc_option_price_rs(s0, k, r, t, 0.0, false, &z);
        let parity = s0 - k * (-r * t).exp();
        assert!((call.price - put.price - parity).abs() < 1e-12);
    }

    /// A single draw cannot estimate variance: error must be zero.
    #[test]
    fn single_draw_has_zero_error() {
        let st = 100.0 * (-0.5 * 0.3f64 * 0.3 + 0.3 * 0.5).exp();
        let expected = (st - 90.0).max(0.0);
        let est = mc_option_price_rs(100.0, 90.0, 0.0, 1.0, 0.3, true, &[0.5]);
        assert!((est.price - expected).abs() < 1e-12);
        assert_eq!(est.error, 0.0);
    }

    /// With a spread of payoffs the standard error must be strictly positive.
    #[test]
    fn error_is_positive_when_payoffs_vary() {
        // z large positive -> deep ITM call; z large negative -> OTM call.
        let z = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let est = mc_option_price_rs(100.0, 100.0, 0.0, 1.0, 0.5, true, &z);
        assert!(est.error > 0.0);
    }

    /// A simple hand-check: r = 0, sigma = 0 => every payoff is the intrinsic
    /// value, so the mean is the intrinsic value and the error is zero.
    #[test]
    fn price_is_mean_of_discounted_payoffs() {
        let est = mc_option_price_rs(100.0, 90.0, 0.0, 1.0, 0.0, true, &[0.0, 0.0]);
        assert!((est.price - 10.0).abs() < 1e-12);
        assert_eq!(est.error, 0.0);
    }
}
