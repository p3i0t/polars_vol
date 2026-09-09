# Installation

## From PyPI

```bash
pip install polars-vol
```

With [uv](https://docs.astral.sh/uv/), either interface works:

```bash
uv add polars-vol            # in a uv-managed project
uv pip install polars-vol    # with the uv pip interface
```

Wheels are published for CPython ≥ 3.9 (abi3, so one wheel covers all
supported versions) on macOS (arm64, x86_64), Linux (x86_64, aarch64) and
Windows (x64). `polars >= 1.3` is installed automatically as a dependency.

## From source

Requires a [Rust toolchain](https://rustup.rs) and
[maturin](https://www.maturin.rs). The repo pins its Rust toolchain in
`rust-toolchain.toml`, so rustup fetches the right one automatically:

```bash
git clone https://github.com/p3i0t/polars_vol && cd polars_vol
pip install maturin          # or: uv pip install maturin
maturin develop --release
```

## Verify the installation

```python
import polars as pl
import polars_vol as pv

df = pl.DataFrame(
    {"strike": [100.0], "spot": [100.0], "vol": [0.2],
     "time_to_expiry": [1.0], "rate": [0.05], "is_call": [True]}
)
print(pv.black_scholes("strike", "spot", "vol", "time_to_expiry", "rate", "is_call"))
# 10.450575  (the textbook value: 10.4506)
```

## Development setup

```bash
make venv             # .venv with the dev dependencies
make install-release  # build and install the Rust extension
make test             # pytest
make pre-commit       # cargo fmt/clippy + ruff + mypy (what CI runs)
make docs             # serve this documentation (mkdocs)
```
