# main.py
"""
Local Forex Educational Signal Bot
- Uses TwelveData for candles
- Indicators: RSI(14), MACD(12,26,9), SMA(9,21)
- Sends educational signals to a single registered Telegram chat
- Auto-scheduler: morning/afternoon/evening quotas and randomized spacing
"""
import os
import asyncio
import logging
import json
import random
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
import pandas_ta as ta
from cachetools import TTLCache
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.constants import ParseMode
# ---------------- Time & Timezone Setup (Nigeria) ----------------

from datetime import datetime, time as dtime, timedelta, date
from pytz import timezone

# Define Nigeria timezone once globally
NIGERIA_TZ = timezone("Africa/Lagos")

def get_nigeria_time() -> datetime:
    """Return the current time in Nigeria timezone."""
    return datetime.now(NIGERIA_TZ)

def now_local_str() -> str:
    """Return the current Nigeria time in a human-readable format (e.g. 2:50 PM)."""
    return get_nigeria_time().strftime("%I:%M %p").lstrip("0")
# ---------------- Load environment variables ----------------
load_dotenv()

# ---------------- Config ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVE_KEY = os.getenv("TWELVEDATA_KEY")

if not TELEGRAM_TOKEN or not TWELVE_KEY:
    raise RuntimeError("Missing TELEGRAM_TOKEN or TWELVEDATA_KEY environment variable")

# Session / timezone config
NIGERIA_TZ = timezone("Africa/Lagos")

# Poll interval for background scheduler (seconds)
POLL_INTERVAL = 20

# Minimum score to consider sending signal (example)
MIN_SCORE_TO_SEND = 2.0  # lower while testing; raise later if needed
# Forex pairs to scan
FOREX_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
    "AUD/USD", "NZD/USD", "USD/CAD", "EUR/GBP"
]

ALLOWED_INTERVALS = ["5min", "15min", "30min", "1h", "4h"]
TWELVE_BASE = "https://api.twelvedata.com/time_series"

# Session definitions (Nigeria local time)
SESSIONS = {
    "morning": {"start": dtime(8, 0), "end": dtime(11, 0), "signals": 5},
    "afternoon": {"start": dtime(12, 45), "end": dtime(15, 0), "signals": 3},
    "evening": {"start": dtime(17, 55), "end": dtime(21, 0), "signals": 5},
}

# Daily quota tracking (for status and preventing duplicates)
STATE_FILE = "bot_state.json"
DEFAULT_STATE = {
    "owner_chat_id": None,
    "date": str(date.today()),
    "counts": {s: 0 for s in SESSIONS.keys()}
}

# Caching candles a little to reduce API calls
OHLC_CACHE = TTLCache(maxsize=400, ttl=18)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("local-forex-bot")

# ---------------- State helpers ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                st = json.load(f)
            # If date changed, reset counts
            if st.get("date") != str(date.today()):
                st["date"] = str(date.today())
                st["counts"] = {s: 0 for s in SESSIONS.keys()}
            return st
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def save_state(st):
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=4)

state = load_state()


# ---------------- Session pair tracking ----------------
SESSION_PAIRS_FILE = "session_pairs.json"

def load_session_pairs():
    if os.path.exists(SESSION_PAIRS_FILE):
        try:
            with open(SESSION_PAIRS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {s: [] for s in SESSIONS.keys()}

def save_session_pairs(data):
    with open(SESSION_PAIRS_FILE, "w") as f:
        json.dump(data, f, indent=4)

session_pairs = load_session_pairs()

def now_local_str() -> str:
    return get_nigeria_time().strftime("%I:%M %p").lstrip("0")

# ---------------- TwelveData fetch ----------------
# ---------------- TwelveData fetch (tz-aware) ----------------
def fetch_twelve_series(symbol: str, interval: str, outputsize: int = 500) -> pd.DataFrame:
    key = f"{symbol}|{interval}"
    if key in OHLC_CACHE:
        return OHLC_CACHE[key].copy()

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_KEY,
        "format": "JSON"
    }
    resp = requests.get(TWELVE_BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("status") == "error":
        raise RuntimeError("TwelveData error: " + str(data.get("message") or data))

    if "values" not in data:
        raise RuntimeError("Unexpected response from TwelveData API")

    rows = data["values"]
    df = pd.DataFrame(rows)

    # parse as UTC then convert index to Nigeria timezone
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.set_index("datetime").sort_index()
    try:
        df.index = df.index.tz_convert(NIGERIA_TZ)
    except Exception:
        pass

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    OHLC_CACHE[key] = df
    return df.copy()

# ---------------- Utility: RSI helper ----------------
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Simple RSI implementation (Wilder's smoothing).
    Returns a pandas Series aligned with series.
    """
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()

    rs = ma_up / (ma_down.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------------- Indicators & scoring ----------------
# ---------------- Indicators & scoring (improved) ----------------
def compute_indicators_and_score(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 50:
        return None

    close = df["close"]

    # Moving Averages (9 & 21)
    sma9 = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()

    # MACD & RSI
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    rsi = compute_rsi(close)

    last = len(close) - 1

    # determine bias
    buy_score = 0
    sell_score = 0
    reasons = []

    # Trend confirmation
    if sma9.iloc[last] > sma21.iloc[last]:
        buy_score += 2
        reasons.append("9/21 SMA bullish crossover")
    elif sma9.iloc[last] < sma21.iloc[last]:
        sell_score += 2
        reasons.append("9/21 SMA bearish crossover")

    # MACD momentum
    if macd_hist.iloc[last] > 0:
        buy_score += 2
        reasons.append(f"MACD hist {macd_hist.iloc[last]:.5f} > 0")
    else:
        sell_score += 2
        reasons.append(f"MACD hist {macd_hist.iloc[last]:.5f} < 0")

    # RSI support
    if rsi.iloc[last] > 60:
        buy_score += 1
        reasons.append(f"RSI {rsi.iloc[last]:.1f} supports BUY")
    elif rsi.iloc[last] < 40:
        sell_score += 1
        reasons.append(f"RSI {rsi.iloc[last]:.1f} supports SELL")
    else:
        reasons.append(f"RSI {rsi.iloc[last]:.1f}")

    # total
    total_score = buy_score - sell_score

    # only return strong setups (no more NEUTRAL)
    if abs(total_score) < 2:
        return None  # reject weak/neutral setups

    signal = "CALL" if total_score > 0 else "PUT"
    return {
        "signal": signal,
        "score": abs(total_score),
        "price": close.iloc[last],
        "reasons": reasons,
        "time": df.index[-1].isoformat()
    }

# ---------------- Telegram Handlers ----------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    txt = (
        f"👋 Hello {user.first_name or 'Trader'}! Welcome to Anthony Forex Educational Bot.\n\n"
        "Here’s what you can do:\n"
        "📊 /register - Subscribe to daily educational signals\n"
        "📈 /board - View current signals\n"
        "❌ /unregister - Stop receiving signals\n"
        "ℹ /status - Check your registration status"
    )
    await update.message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)

async def register_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global state
    state["owner_chat_id"] = update.message.chat_id
    save_state(state)
    await update.message.reply_text("✅ This chat is now registered. You will receive signals here.")

async def unregister_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global state
    if state.get("owner_chat_id") == update.message.chat_id:
        state["owner_chat_id"] = None
        save_state(state)
        await update.message.reply_text("❌ Unregistered successfully.")
    else:
        await update.message.reply_text("This chat is not the registered owner.")

async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    now = get_nigeria_time()
    current_time = now.strftime("%I:%M %p")
    current_date = now.strftime("%Y-%m-%d")

    parts = [
        f"📅 Date: {current_date}",
        f"🕒 Time now: {current_time} (Nigeria)",
        "Today's counts:"
    ]
    for s, cfg in SESSIONS.items():
        parts.append(f"- {s.title()}: {state['counts'].get(s, 0)} / {cfg['signals']}")

    await update.message.reply_text("\n".join(parts))
# ---------------- Board Handler ----------------
async def board_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show top current setups from indicator analysis."""
    await update.message.reply_text("🕵 Building board (please wait a few seconds)...")

    topn = []
    for symbol in FOREX_PAIRS:
        try:
            df = fetch_twelve_series(symbol, "15min")
            info = compute_indicators_and_score(df)
            if info:
                topn.append({"pair": symbol, **info})
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            # keep going to next symbol

    if not topn:
        await update.message.reply_text("No setups available right now.")
        return

    # sort by score descending
    topn = sorted(topn, key=lambda x: x["score"], reverse=True)

    now = get_nigeria_time()
    date_now_str = now.strftime("%Y-%m-%d")
    time_now_str = now.strftime("%I:%M %p")

    lines = [
        f"📅 Date: {date_now_str}",
        f"🕒 Time now: {time_now_str} (Nigeria)",
        "",
        "Top setups"
    ]

    nigeria_tz = timezone("Africa/Lagos")

    for i, c in enumerate(topn, start=1):
        try:
            dt = datetime.fromisoformat(c["time"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone("UTC"))
            local_time = dt.astimezone(nigeria_tz).strftime("%I:%M %p")
        except Exception:
            local_time = "N/A"

        lines.append(f"{i}. {c['pair']} — {c['signal']} (score {c['score']:.1f})")
        lines.append(f"💰 Price: {float(c['price']):.5f} | 🕒 Time: {local_time} (Nigeria)")
        if c.get("reasons"):
            lines.append("• " + " | ".join(c.get("reasons", [])))
        lines.append("")

    await update.message.reply_text("\n".join(lines))
async def send_signal(app, session_name: str, number: int):
    """Run the analysis and send one educational signal to the registered owner."""
    owner = state.get("owner_chat_id")

    # --- Step 5B: Avoid repeating pairs within a session ---
    used_pairs = session_pairs.get(session_name, [])
    available_pairs = [p for p in FOREX_PAIRS if p not in used_pairs]
    if not available_pairs:
        logger.info("All pairs already used for %s; resetting.", session_name)
        available_pairs = FOREX_PAIRS.copy()
        session_pairs[session_name] = []
        save_session_pairs(session_pairs)

    if not owner:
        logger.warning("No owner registered; skipping send")
        return

    # pick best pair at current moment (scan multiple timeframes)
    candidates = []
    TIMEFRAMES = ["5min", "15min", "30min", "1h"]

    for symbol in available_pairs:
        best_info = None
        for tf in TIMEFRAMES:
            try:
                df = fetch_twelve_series(symbol, tf)
                info = compute_indicators_and_score(df)
                if not info:
                    continue
                if info.get("score", 0) < MIN_SCORE_TO_SEND:
                    continue
                if not best_info or info["score"] > best_info["score"]:
                    info["timeframe"] = tf
                    best_info = info
            except Exception as e:
                logger.debug("Analysis error for %s (%s): %s", symbol, tf, e)
                continue
        if best_info:
            candidates.append({"pair": symbol, **best_info})

    if not candidates:
        logger.info("No candidates to send for %s session", session_name)
        return

    top = sorted(candidates, key=lambda x: x["score"], reverse=True)[0]

    # Mark the pair as used for this session
    try:
        if top["pair"] not in session_pairs.get(session_name, []):
            session_pairs.setdefault(session_name, []).append(top["pair"])
            save_session_pairs(session_pairs)
    except Exception as e:
        logger.debug("Failed to mark pair used: %s", e)

    # build message
    entry_time = (get_nigeria_time() + timedelta(minutes=1)).strftime("%I:%M %p")
    msg_lines = [
        "📊 Educational Signal (Live Market)",
        f"🕒 Time now: {now_local_str()}",
        f"📈 Session: {session_name.title()}",
        f"• Pair: {top['pair']}",
        f"• Direction: {top['signal']}",
        f"• Entry Time: {entry_time}",
        f"• Timeframe: {top.get('timeframe', '15min')}",
        f"• Reasoning: " + ("; ".join(top.get('reasons', [])) or "Analysis"),
        "",
        "This is educational only — test on demo before risking real capital."
    ]

    try:
        await app.bot.send_message(chat_id=owner, text="\n".join(msg_lines), parse_mode=ParseMode.MARKDOWN)
        # update today's counts
        for s, cfg in SESSIONS.items():
            if s == session_name:
                state["counts"][s] = state["counts"].get(s, 0) + 1
                save_state(state)
                break
        logger.info("Sent educational signal: %s %s", top["pair"], session_name)
    except Exception as e:
        logger.exception("Failed to send signal: %s", e)

async def session_scheduler(app):
    """Continuously check time and run sessions when active. Designed to run as background task."""
    logger.info("Scheduler started (Nigeria TZ)")
    while True:
        logger.debug("Scheduler tick: now=%s counts=%s", get_nigeria_time().isoformat(), state.get("counts"))
        now = get_nigeria_time()
        tnow = now.time()
        for name, cfg in SESSIONS.items():
            start_t = cfg["start"]
            end_t = cfg["end"]
            quota = cfg["signals"]
            # If current time inside session window and we haven't reached quota
            if start_t <= tnow <= end_t and state["counts"].get(name, 0) < quota:
                logger.info("Session active: %s (count %s/%s)", name, state['counts'].get(name, 0), quota)
                # compute total minutes and per-signal spacing
                total_minutes = (datetime.combine(now.date(), end_t) - datetime.combine(now.date(), start_t)).seconds / 60
                per_signal = max(1, total_minutes / quota)
                # send remaining signals for the session, spaced randomly
                remaining = quota - state["counts"].get(name, 0)
                for i in range(remaining):
                    # re-check if still in session window
                    now2 = get_nigeria_time()
                    if now2.time() > end_t:
                        break
                    await send_signal(app, name, state["counts"].get(name, 0) + 1)
                    wait_minutes = random.uniform(per_signal * 0.5, per_signal * 1.2)
                    logger.info("Waiting %.1f minutes before next signal in %s", wait_minutes, name)
                    await asyncio.sleep(wait_minutes * 60)
        await asyncio.sleep(60)  # check each minute

# ---------------- Main runner ----------------
async def run_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("register", register_handler))
    app.add_handler(CommandHandler("unregister", unregister_handler))
    app.add_handler(CommandHandler("status", status_handler))
    app.add_handler(CommandHandler("board", board_handler))

    logger.info("🚀 Forex educational bot is running...")

    # Start both the bot and the scheduler concurrently
    async with app:
        asyncio.create_task(session_scheduler(app))
        await app.run_polling()


if __name__ == "__main__":
    import asyncio
    import nest_asyncio

    nest_asyncio.apply()  # Allow nested loops safely

    try:
        asyncio.run(run_bot())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Bot stopped manually.")
