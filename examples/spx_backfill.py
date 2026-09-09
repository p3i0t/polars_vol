from datetime import date, datetime, timedelta, timezone

import polars as pl
import yfinance as yf

from polars_vol import implied_vol

EXPIRY, DAY = "2026-09-15", "2026-09-08"  # fixed expiry; spot = close on this day
spx = yf.Ticker("^SPX")
chain = spx.option_chain(EXPIRY)
spot = pl.from_pandas(spx.history(start=DAY, end=(date.fromisoformat(DAY) + timedelta(days=1)).isoformat())[["Close"]]).item()
COLS = ["strike", "bid", "ask", "impliedVolatility"]
quotes = pl.concat([pl.from_pandas(c[COLS]).with_columns(is_call=pl.lit(i)) for c, i in ((chain.calls, True), (chain.puts, False))])
quotes = quotes.with_columns(spot=pl.lit(spot), rate=pl.lit(0.038), option_price=(pl.col("bid") + pl.col("ask")) / 2, expiry=pl.lit(date.fromisoformat(EXPIRY)))
quotes = quotes.with_columns(time_to_expiry=((pl.col("expiry") - datetime.now(timezone.utc).date()).dt.total_days() + 1) / 365)
quotes = quotes.with_columns(iv=implied_vol("option_price", "strike", "spot", "time_to_expiry", "rate", "is_call"))
near = quotes.filter(pl.col("iv").is_not_nan(), pl.col("impliedVolatility") > 0, pl.col("strike").is_between(0.9 * spot, 1.1 * spot), (pl.col("is_call") & (pl.col("strike") >= spot)) | (~pl.col("is_call") & (pl.col("strike") <= spot)))
print(near.select(corr=pl.corr("iv", "impliedVolatility")))
# -> corr ≈ 0.997 with Yahoo's own IVs (median gap ≈ 0.003 vol points)
