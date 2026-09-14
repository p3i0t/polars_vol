"""Minimal end-to-end example: CFFEX index-option chain -> IV + Greeks.

For one trading day, fetch the full CSI 300 / CSI 500 / SSE 50 index-option
chain from CFFEX's daily XML and the underlying index closes via AkShare,
merge them into a single Polars DataFrame, then backfill Black-Scholes
implied volatility and Greeks with a single polars-vol expression.

Run:  python examples/csi_index_options_backfill.py [YYYYMMDD]
"""

import calendar
import sys
from datetime import date
from xml.etree import ElementTree

import akshare as ak
import polars as pl
import requests
import polars_vol as pv

UNDERLYING = {
    "IO": "sh000300",
    "MO": "sh000905",
    "HO": "sh000016",
}  # -> AkShare symbols
RISK_FREE_RATE = 0.02

INSTRUMENT_RE = (
    r"^(?P<product>[A-Z]+)(?P<yymm>\d{4})-(?P<ctype>[CP])-(?P<strike>\d+(?:\.\d+)?)$"
)


def fetch_chain(day: str) -> pl.DataFrame:
    """One day of CFFEX daily quotes, index options only."""
    root = ElementTree.fromstring(
        requests.get(
            f"http://www.cffex.com.cn/sj/hqsj/rtj/{day[:6]}/{day[6:]}/index.xml",
            headers={
                "Referer": "http://www.cffex.com.cn/en_new/DailyData.html",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=30,
        ).content
    )
    rows = [
        {
            "instrument_id": (e.findtext("instrumentid") or "").strip(),
            "product_id": (e.findtext("productid") or "").strip(),
            "expire_date": e.findtext("expiredate"),
            "close": float(e.findtext("closeprice") or 0),
            "settlement": float(e.findtext("settlementprice") or 0),
            "volume": int(e.findtext("volume") or 0),
            "open_interest": int(e.findtext("openinterest") or 0),
        }
        for e in root.findall("dailydata")
        if (e.findtext("productid") or "").strip() in UNDERLYING
    ]
    return pl.DataFrame(rows)


def fetch_spot(day: str) -> pl.DataFrame:
    """Underlying index closes for the same day, one row per product."""
    frames = (
        pl.from_dataframe(ak.stock_zh_index_daily(symbol))
        .filter(pl.col("date") == date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}"))
        .select(
            pl.lit(product).alias("product_id"),
            pl.col("close").alias("underlying_close"),
        )
        for product, symbol in UNDERLYING.items()
    )
    return pl.concat(frames)


def third_friday(yymm: str) -> date:
    """CFFEX index options expire on the 3rd Friday of the contract month."""
    year, month = 2000 + int(yymm[:2]), int(yymm[2:4])
    first_friday = 1 + (4 - calendar.monthrange(year, month)[0]) % 7
    return date(year, month, first_friday + 14)


def prepare_df(day: str) -> pl.DataFrame:
    """Parse contract fields, attach time to expiry, solve IV + Greeks."""
    df = fetch_chain(day).join(fetch_spot(day), on="product_id")
    trading_day = date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}")
    return (
        df.select(
            pl.all(),
            pl.col("instrument_id").str.extract_groups(INSTRUMENT_RE).alias("p"),
        )
        .unnest("p")
        .with_columns(
            strike=pl.col("strike").cast(pl.Float64),
            trading_day=pl.lit(trading_day),
        )
        .with_columns(
            expire_date_eff=pl.col("yymm").map_elements(
                third_friday, return_dtype=pl.Date
            ),
        )
        .with_columns(
            time_to_expiry=(
                pl.col("expire_date_eff") - pl.col("trading_day")
            ).dt.total_days()
            / 365.0,
        )
    )


def plot(df_backfill: pl.DataFrame):
    import plotly.graph_objects as go

    d = df_backfill.filter(df_backfill["implied_vol"].is_not_nan())
    d = d.with_columns(
        k=(
            d["strike"]
            / (d["underlying_close"] * (2.718281828 ** (0.02 * d["time_to_expiry"])))
        ).log()
    )

    # One plot per underlying: IO / MO / HO are different indices with
    # different price levels, so their smiles live on different scales.
    for product in d["product_id"].unique().sort().to_list():
        m = d.filter(pl.col("product_id") == product)
        # Standard OTM smile: OTM calls (K >= F, k >= 0) joined with OTM puts
        # (K < F, k < 0), merged and sorted by log-moneyness.
        otm_call = m.filter((pl.col("ctype") == "C") & (pl.col("k") >= 0.0))
        otm_put = m.filter((pl.col("ctype") == "P") & (pl.col("k") < 0.0))
        smile = pl.concat([otm_call, otm_put])

        fig = go.Figure()
        # One connected curve per expiry; call and put sides are marked
        # with different colours and markers.
        for tte in smile["time_to_expiry"].unique().sort().to_list():
            g = smile.filter(pl.col("time_to_expiry") == tte).sort("k")
            for side, (ctype, color, symbol) in (
                ("put", ("P", "#1f77b4", "circle")),
                ("call", ("C", "#d62728", "diamond")),
            ):
                s = g.filter(pl.col("ctype") == ctype)
                fig.add_trace(
                    go.Scatter3d(
                        x=s["k"],
                        y=s["time_to_expiry"] * 365,
                        z=s["implied_vol"],
                        mode="lines+markers",
                        line=dict(width=2, color=color),
                        marker=dict(size=3, color=color, symbol=symbol),
                        name=f"{tte * 365:.0f}d · {side}",
                    )
                )
        fig.update_layout(
            title=f"{product} — OTM implied volatility smile ({day})",
            scene=dict(
                xaxis_title="log-moneyness  ln(K / F)",
                yaxis_title="days to expiry",
                zaxis_title="implied volatility",
            ),
            legend_title_text="expiry · side",
        )
        fig.write_html(f"examples/csi_index_options_backfill_{day}_{product}.html")
        print(
            f"wrote examples/csi_index_options_backfill_{day}_{product}.html "
            f"({smile.height} OTM contracts)"
        )


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else "20260901"
    df = prepare_df(day)
    print(df)

    df_backfill = (
        df.select(
            "instrument_id",
            "product_id",
            "settlement",
            "underlying_close",
            "time_to_expiry",
            "ctype",
            "strike",
            backfill=pv.iv_and_greeks(
                "settlement",
                "strike",
                "underlying_close",
                "time_to_expiry",
                pl.lit(RISK_FREE_RATE),
                pl.col("ctype") == "C",
            ),
        )
        .unnest("backfill")
        .with_columns(  # canonical units -> market quoting conventions
            vega=pl.col("vega") / 100,  # per 1 vol point
            theta=pl.col("theta") / 365,  # per calendar day
        )
        .rename({"delta": "delta_bs", "model_price": "bs_option_price"})
    ).filter(pl.col("implied_vol").is_finite())

    print(
        f"{day}: {df_backfill.height} contracts, {df_backfill.filter(pl.col('implied_vol').is_finite()).height} with solved IV"
    )
    pl.Config.set_tbl_cols(15)
    print(df_backfill.head(10))
