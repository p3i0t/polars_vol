SHELL=/bin/bash

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

install:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop

install-release:
	unset CONDA_PREFIX && \
	source .venv/bin/activate && maturin develop --release

pre-commit:
	cargo fmt --all && cargo clippy --all-features -- -D warnings
	.venv/bin/python -m ruff check . --fix --exit-non-zero-on-fix
	.venv/bin/python -m ruff format polars_vol tests examples
	.venv/bin/python -m mypy polars_vol tests

test:
	.venv/bin/python -m pytest tests

test-rust:
	DYLD_LIBRARY_PATH=$$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))') \
		cargo test --no-default-features

demo: install-release
	source .venv/bin/activate && PYTHONPATH=. python examples/speed_vs_numpy.py

spx: install-release
	source .venv/bin/activate && PYTHONPATH=. python examples/spx_backfill.py

docs:
	python -m mkdocs serve
