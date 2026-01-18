import os
import sys
import argparse
import re
import random
import time
import json
import mimetypes
import requests
import tweepy
from copy import deepcopy
from tweepy.errors import Forbidden, TooManyRequests, TweepyException
from google import genai
from google.genai import types
import shutil
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import feedparser

plt.switch_backend('Agg')

# ================= INITIAL SETUP =================

load_dotenv()
external_env = os.getenv('EXTERNAL_ENV_FILE')
if external_env and os.path.exists(external_env):
    load_dotenv(external_env, override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=5 * 1024 * 1024, backupCount=2, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================

COINGECKO_URL = os.getenv(
    'COINGECKO_URL',
    'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,fractal-bitcoin&vs_currencies=usd&precision=full'
)
BACKEND_URL = os.getenv('BACKEND_URL', 'https://fennec-api.warninghejo.workers.dev')
TICKER_FENNEC = os.getenv('TICKER_FENNEC', 'FENNEC')
TICKER_FB = os.getenv('TICKER_FB', 'sFB___000')
UNISAT_API_KEY = os.getenv('UNISAT_API_KEY', '')
UNISAT_BASE_URL = os.getenv('UNISAT_BASE_URL', 'https://open-api-fractal.unisat.io')

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
ADMIN_USER_ID = os.getenv('ADMIN_USER_ID', '')

DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))
CHART_COOLDOWN = 6 * 3600  # 6 hours
PRICE_TRIGGER_PCT = 5.0
GLOBAL_POST_INTERVAL = 3600  # 1 hour cooldown for proactive posts
MIN_POST_INTERVAL = int(os.getenv('MIN_POST_INTERVAL', '900'))
MAX_POST_INTERVAL = int(os.getenv('MAX_POST_INTERVAL', '7200'))
VOLATILITY_FAST_THRESHOLD = float(os.getenv('VOLATILITY_FAST_THRESHOLD', '15'))
VOLATILITY_SLOW_THRESHOLD = float(os.getenv('VOLATILITY_SLOW_THRESHOLD', '5'))
NIGHT_COOLDOWN_MULTIPLIER = float(os.getenv('NIGHT_COOLDOWN_MULTIPLIER', '1.5'))
NIGHT_START_HOUR = int(os.getenv('NIGHT_START_HOUR', '0'))
NIGHT_END_HOUR = int(os.getenv('NIGHT_END_HOUR', '6'))
MAX_TWEET_LENGTH = int(os.getenv('MAX_TWEET_LENGTH', '280'))
TWEET_SAFETY_BUFFER = int(os.getenv('TWEET_SAFETY_BUFFER', '4'))
BASE_BACKOFF_SECONDS = 15 * 60  # 15 minutes
MAX_BACKOFF_SECONDS = 4 * 60 * 60  # 4 hours
MAX_CONSECUTIVE_ERRORS = 3
ERROR_WINDOW_SECONDS = 3600

# 2026 Model Configuration
# Primary: The current speed flagship
MODEL_PRIMARY = os.getenv('MODEL_PRIMARY', 'gemini-3-flash')
# Backup: The unlimited lite version
MODEL_BACKUP = os.getenv('MODEL_BACKUP', 'gemini-2.5-flash-lite')

# News Configuration
RSS_FEEDS = [
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
]
NEWS_KEYWORDS = ['bitcoin', 'crypto', 'sec', 'etf', 'regulation', 'market']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_FILE_PATH = os.path.join(BASE_DIR, 'STOPPED_BY_ERROR')
STATE_ARCHIVE_DIR = os.path.join(BASE_DIR, 'state_archive')
DIR_BURN = os.path.join(BASE_DIR, 'images_burn')
DIR_GM = os.path.join(BASE_DIR, 'images_gm')
DIR_REGULAR = os.path.join(BASE_DIR, 'images_regular')
DIR_USED = os.path.join(BASE_DIR, 'images_used')
STATE_FILE = os.path.join(BASE_DIR, 'state.json')
HEALTHCHECK_FILE = os.getenv('HEALTHCHECK_FILE', os.path.join(BASE_DIR, 'healthcheck.txt'))

COINGECKO_CACHE_TTL = int(os.getenv('COINGECKO_CACHE_TTL', '120'))
POOL_CACHE_TTL = int(os.getenv('POOL_CACHE_TTL', '60'))
MAX_STATE_BACKUPS = int(os.getenv('MAX_STATE_BACKUPS', '5'))
ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')
OFFLINE_MODE = os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
WATCHDOG_PING_ENABLED = os.getenv('WATCHDOG_PING_ENABLED', 'true').lower() == 'true'
WATCHDOG_PING_URL = os.getenv('WATCHDOG_PING_URL', '')
WATCHDOG_PING_METHOD = os.getenv('WATCHDOG_PING_METHOD', 'post').lower()
WATCHDOG_PING_HEADERS = os.getenv('WATCHDOG_PING_HEADERS', '')
TELEGRAM_COMMAND_SECRET = os.getenv('TELEGRAM_COMMAND_SECRET', '')
OFFLINE_MARKET_FILE = os.getenv('OFFLINE_MARKET_FILE', os.path.join(BASE_DIR, 'offline_market.json'))

try:
    WATCHDOG_HEADERS = json.loads(WATCHDOG_PING_HEADERS) if WATCHDOG_PING_HEADERS else {}
except json.JSONDecodeError:
    WATCHDOG_HEADERS = {}

_extra_ids = [i.strip() for i in os.getenv('TELEGRAM_ALLOWED_IDS', '').split(',') if i.strip()]
_base_ids = [str(ADMIN_USER_ID), str(TELEGRAM_CHAT_ID)]
TELEGRAM_ALLOWED_IDS = {cid for cid in _base_ids + _extra_ids if cid}

REGULAR_POST_HOURS = [11, 15, 19, 23]
GM_HOURS = [7, 8, 9, 10]

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

HEADERS = {
    'User-Agent': os.getenv(
        'HTTP_USER_AGENT',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json'
}


def _get_cached_payload(state, key, ttl):
    cache = state.get('cache', {})
    payload = cache.get(key)
    ts = cache.get(f"{key}_ts", 0)
    if payload and (time.time() - ts) < ttl:
        return payload
    return None


def _store_cache(state, key, value):
    cache = state.setdefault('cache', {})
    cache[key] = value
    cache[f"{key}_ts"] = time.time()

PERSONA = (
    "You are Fennec, a cyber-fox AI running on Fractal Bitcoin. "
    "You analyze CoinGecko prices and local liquidity pools. "
    "Personality: Crypto-native, witty, bullish but realistic, cyberpunk vibes. "
    "\n\nGUIDELINES:"
    "\n- MAIN POSTS: Always in ENGLISH."
    "\n- FORMATTING: Use Unicode Bold for numbers. Always use '$' for prices."
    "\n- BULLETS: Use '🔸' as the bullet point symbol."
    "\n- PRECISION: For Fractal (FB), use 5 decimal places (e.g., $0.40321). Do not round too much."
    "\n- TAG STRUCTURE: Finish the tweet body, then add two newlines (\\n\\n) followed by a single line of tags."
    "\n- TAG MIX: Tag line must include exactly 4-6 tags in one line separated by spaces."
    "\n- REQUIRED TAGS: Always include $FENNEC, $FB, and #FractalBitcoin in the tag line."
    "\n- SYMBOL RULES: Use '$' ONLY for tickers (e.g., $FB, $BTC, $FENNEC). Use '#' ONLY for topics (e.g., #FractalBitcoin, #DeFi). Never mix."
    "\n- OPTIONAL TAGS: Add 1-2 relevant topical hashtags (e.g., #Mining, #Memecoin, #CAT20) if they match the content."
    "\n- EXAMPLE TAG LINE: $FENNEC $FB #FractalBitcoin #DeFi"
    "\n- NEVER place tags inside the tweet body—only in the closing tag line."
    "\n- NO MARKDOWN: Never use asterisks (**) or other formatting symbols."
    "\n- LENGTH: ALL tweets must be under 280 characters."
)

DEFAULT_STATE = {
    'burn_day_counter': 201,
    'last_gm_date': '',
    'last_regular_post_hour': -1,
    'last_update_id': 0,
    'price_history': [],
    'last_alert_price': 0.0,
    'last_alert_time': 0,
    'last_known_prices': {},
    'last_news_link': '',
    'last_any_post_time': 0,
    'last_chart_post_time': 0,
    'error_events': [],
    'last_health_report_date': '',
    'last_state_snapshot': '',
    'dynamic_cooldown_seconds': GLOBAL_POST_INTERVAL,
    'prophecy': {
        'last_run_date': '',
        'last_tweet_id': None,
        'start_price': 0.0
    },
    'cache': {
        'coingecko': None,
        'coingecko_ts': 0,
        'pool': None,
        'pool_ts': 0
    }
}


def _format_price(value, fb_precision=False):
    if fb_precision:
        formatted = f"${value:.5f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted if formatted != "$" else "$0.00"
    
    if value >= 1000:
        return f"${value:,.0f}"
    if value >= 1:
        return f"${value:,.2f}"
    if value >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _build_system_config():
    return types.GenerateContentConfig(system_instruction=PERSONA)


def _call_genai(client, contents):
    if not client:
        return None

    if isinstance(contents, str):
        payload = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=contents)]
            )
        ]
    else:
        payload = contents

    for model_name in [MODEL_PRIMARY, MODEL_BACKUP]:
        try:
            if DEBUG_MODE:
                logger.info(f"🧠 Thinking with {model_name}...")

            config = _build_system_config()
            current_model = model_name.replace("models/", "")
            response = client.models.generate_content(
                model=current_model,
                contents=payload,
                config=config
            )

            text = getattr(response, 'text', None)
            if text:
                return text.strip().replace('"', '').replace("'", "")

        except Exception as exc:
            logger.warning(f"⚠️ Model {model_name} failed: {exc}")
            continue

    logger.error("❌ All AI models (Gemini 3 & 2.5) failed to respond.")
    return None


def _load_image_part(image_path):
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = 'image/png'
        with open(image_path, 'rb') as img_file:
            data = img_file.read()
        return types.Part.from_bytes(data=data, mime_type=mime_type)
    except Exception as exc:
        logger.error(f"Image Load Error: {exc}")
        return None

# ================= UTILITIES =================

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sanitize_reason(reason: str) -> str:
    if not reason:
        return "auto"
    safe = ''.join(ch for ch in reason if ch.isalnum() or ch in ('-', '_'))
    return safe[:40] or "auto"


def _backup_state_file(reason="auto"):
    if reason == "auto" or not os.path.exists(STATE_FILE):
        return
    try:
        _ensure_dir(STATE_ARCHIVE_DIR)
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        label = _sanitize_reason(reason)
        backup_path = os.path.join(STATE_ARCHIVE_DIR, f"state_{ts}_{label}.json")
        shutil.copy2(STATE_FILE, backup_path)
        backups = sorted(Path(STATE_ARCHIVE_DIR).glob('state_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[MAX_STATE_BACKUPS:]:
            try:
                old.unlink()
            except Exception:
                logger.warning(f"Unable to delete old state backup: {old}")
    except Exception as e:
        logger.error(f"State backup error: {e}")


def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {}
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        data = {}
    merged = deepcopy(DEFAULT_STATE)
    merged.update(data)
    return merged


def save_state(state, reason="auto"):
    try:
        _ensure_dir(os.path.dirname(STATE_FILE))
        if reason != "auto":
            _backup_state_file(reason=reason)
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving state: {e}")


def reset_state(reason="manual_reset"):
    logger.warning(f"State reset triggered ({reason})")
    fresh_state = deepcopy(DEFAULT_STATE)
    save_state(fresh_state, reason=reason)
    return fresh_state


def archive_state(reason="manual_archive"):
    if not os.path.exists(STATE_FILE):
        logger.warning("Archive skipped: state file missing")
        return False
    _backup_state_file(reason=reason)
    logger.info(f"State archived ({reason})")
    return True


def maybe_snapshot_state(state):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if state.get('last_state_snapshot') == today:
        return state
    try:
        _backup_state_file(reason=f"snapshot_{today}")
        state['last_state_snapshot'] = today
        save_state(state)
    except Exception as exc:
        logger.error(f"Daily snapshot failed: {exc}")
    return state


def _format_timestamp(ts):
    if not ts:
        return "never"
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _state_overview(state):
    lines = [
        f"🔥 Burn Day: {state.get('burn_day_counter')}",
        f"🕒 Last post: {_format_timestamp(state.get('last_any_post_time'))}",
        f"📈 Last chart: {_format_timestamp(state.get('last_chart_post_time'))}",
        f"🌅 Last GM: {state.get('last_gm_date') or 'never'}",
        f"📊 Price history points: {len(state.get('price_history', []))}",
        f"⚠️ Error streak: {len(state.get('error_events', []))}/{MAX_CONSECUTIVE_ERRORS}",
        f"📂 Last snapshot: {state.get('last_state_snapshot') or 'never'}",
        f"⏱ Cooldown: {state.get('dynamic_cooldown_seconds', GLOBAL_POST_INTERVAL)}s"
    ]
    return "\n".join(lines)


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Fennec Bot Controller")
    parser.add_argument('--reset-state', action='store_true', help='Reset state.json to defaults (with backup).')
    parser.add_argument('--archive-state', action='store_true', help='Create a manual archive copy of state.json.')
    parser.add_argument('--snapshot-state', action='store_true', help='Force a daily snapshot entry.')
    parser.add_argument('--show-state', action='store_true', help='Print current state overview and exit.')
    return parser.parse_args()


def _authorize_command(raw_text, chat_id):
    if not TELEGRAM_COMMAND_SECRET:
        return True, raw_text
    parts = raw_text.split()
    if len(parts) < 2:
        logger.warning(f"Command auth missing secret from chat {chat_id}")
        return False, raw_text
    secret = parts[1]
    if secret != TELEGRAM_COMMAND_SECRET:
        logger.warning(f"Command auth failed for chat {chat_id}")
        return False, raw_text
    cleaned = " ".join([parts[0]] + parts[2:])
    return True, cleaned.strip()


def _emit_alert_webhook(event_type, message, extra=None):
    if not ALERT_WEBHOOK_URL:
        return
    payload = {
        'event': event_type,
        'message': message,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'extra': extra or {}
    }
    try:
        requests.post(ALERT_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as exc:
        logger.error(f"Alert webhook failed: {exc}")


def _load_offline_market_snapshot():
    if not os.path.exists(OFFLINE_MARKET_FILE):
        return None
    try:
        with open(OFFLINE_MARKET_FILE, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return {
            'btc': float(data.get('btc', 0.0)),
            'fb': float(data.get('fb', 0.0)),
            'fennec': float(data.get('fennec', 0.0)),
            'fennec_in_fb': float(data.get('fennec_in_fb', 0.0))
        }
    except Exception as exc:
        logger.error(f"Offline market file error: {exc}")
        return None


def _fetch_coingecko_snapshot(state):
    cached = _get_cached_payload(state, 'coingecko', COINGECKO_CACHE_TTL)
    if cached:
        return cached
    try:
        cg_resp = requests.get(COINGECKO_URL, headers=HEADERS, timeout=10)
        cg_resp.raise_for_status()
        cg_data = cg_resp.json()
        snapshot = {
            'btc': float(cg_data.get('bitcoin', {}).get('usd', 0.0)),
            'fb': float(cg_data.get('fractal-bitcoin', {}).get('usd', 0.0))
        }
        _store_cache(state, 'coingecko', snapshot)
        return snapshot
    except Exception as exc:
        logger.error(f"CoinGecko Failed: {exc}")
        return None


def _fetch_pool_snapshot(state, fb_usd):
    cached = _get_cached_payload(state, 'pool', POOL_CACHE_TTL)
    if cached:
        return cached
    try:
        quote_resp = requests.get(
            f"{BACKEND_URL}?action=quote&tick0={TICKER_FENNEC}&tick1={TICKER_FB}",
            headers=HEADERS,
            timeout=10
        )
        quote_resp.raise_for_status()
        pool_data = quote_resp.json().get('data', {})
        amt0 = float(pool_data.get('amount0', 0))
        amt1 = float(pool_data.get('amount1', 0))
        snapshot = {
            'amount0': amt0,
            'amount1': amt1,
            'fennec_in_fb': (amt1 / amt0) if amt0 else 0.0,
            'fennec_usd': ((amt1 / amt0) * fb_usd) if amt0 and fb_usd else 0.0
        }
        _store_cache(state, 'pool', snapshot)
        return snapshot
    except Exception as exc:
        logger.error(f"Pool API Failed: {exc}")
        return None


def _is_night_hour(hour):
    if NIGHT_START_HOUR <= NIGHT_END_HOUR:
        return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR
    return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR


def _compute_dynamic_cooldown(changes):
    cooldown = GLOBAL_POST_INTERVAL
    fen_move = abs((changes or {}).get('fennec') or 0.0)
    if fen_move >= VOLATILITY_FAST_THRESHOLD:
        cooldown = max(MIN_POST_INTERVAL, GLOBAL_POST_INTERVAL * 0.5)
    elif fen_move <= VOLATILITY_SLOW_THRESHOLD:
        cooldown = min(MAX_POST_INTERVAL, GLOBAL_POST_INTERVAL * 1.4)
    cooldown = max(MIN_POST_INTERVAL, min(MAX_POST_INTERVAL, cooldown))
    hour = datetime.now().hour
    if _is_night_hour(hour):
        cooldown = min(MAX_POST_INTERVAL, int(cooldown * NIGHT_COOLDOWN_MULTIPLIER))
    return int(cooldown)


def enforce_tweet_length(text):
    if not text:
        return text
    limit = max(16, MAX_TWEET_LENGTH - TWEET_SAFETY_BUFFER)
    working = re.sub(r'[ \t]{2,}', ' ', text.strip())
    working = re.sub(r'\n{3,}', '\n\n', working)
    if len(working) <= limit:
        return working

    logger.warning(f"Tweet too long ({len(working)} chars). Trimming.")
    lines = working.split('\n')
    while len('\n'.join(lines).strip()) > limit and len(lines) > 1:
        lines = lines[:-1]
    working = '\n'.join(lines).strip()

    if len(working) > limit:
        working = working[:limit - 1].rstrip()
        working = working[:-1] + '…' if working and working[-1].isalnum() and len(working) >= limit else working + '…'

    if len(working) < 16:
        logger.error("Tweet trimming left message unusable.")
        return None

    return working

def send_telegram(text, image_path=None, chat_id=None):
    target_id = chat_id if chat_id else TELEGRAM_CHAT_ID
    if not TELEGRAM_BOT_TOKEN or not target_id:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
        if image_path:
            with open(image_path, 'rb') as f:
                requests.post(url + 'sendPhoto', data={'chat_id': target_id, 'caption': text[:1000]}, files={'photo': f}, timeout=10)
        else:
            requests.post(url + 'sendMessage', data={'chat_id': target_id, 'text': text}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram Error: {e}")


def _extract_status_code(error):
    response = getattr(error, 'response', None)
    if response is not None and hasattr(response, 'status_code'):
        return response.status_code
    return getattr(error, 'status_code', None)


def _compute_backoff_seconds(attempt):
    return min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)


def _record_error_event(state, reason=None):
    if state is None:
        return
    now = time.time()
    windowed = [ts for ts in state.get('error_events', []) if now - ts <= ERROR_WINDOW_SECONDS]
    windowed.append(now)
    state['error_events'] = windowed
    save_state(state)
    extra = {
        'count': len(windowed),
        'window_seconds': ERROR_WINDOW_SECONDS,
        'threshold': MAX_CONSECUTIVE_ERRORS,
        'reason': reason
    }
    if len(windowed) >= MAX_CONSECUTIVE_ERRORS:
        msg = reason or "Exceeded consecutive error threshold"
        _emit_alert_webhook('circuit_breaker_pretrigger', msg, extra)
        _trigger_circuit_breaker(msg)
    elif len(windowed) == MAX_CONSECUTIVE_ERRORS - 1:
        _emit_alert_webhook('error_warning', 'Error streak approaching circuit breaker', extra)


def _reset_error_events(state):
    if state and state.get('error_events'):
        state['error_events'] = []
        save_state(state)


def _trigger_circuit_breaker(reason):
    message = f"{datetime.utcnow().isoformat()}Z — {reason}"
    try:
        Path(STOP_FILE_PATH).write_text(message, encoding='utf-8')
    except Exception as exc:
        logger.error(f"Failed to write circuit breaker file: {exc}")
    send_telegram(f"🛑 BOT AUTO-SHUTDOWN\n{message}")
    _emit_alert_webhook('circuit_breaker', message)
    logger.critical(f"Circuit breaker engaged: {reason}")
    raise SystemExit(reason)

# ================= MARKET DATA LOGIC =================

def update_price_history(state, snapshot):
    history = state.get('price_history', [])
    timestamp = int(time.time())
    entry = {'timestamp': timestamp, **snapshot}
    history.append(entry)
    cutoff = timestamp - 86400
    state['price_history'] = [point for point in history if point.get('timestamp', 0) >= cutoff]

def calculate_trend(state):
    history = state.get('price_history', [])
    if len(history) < 2:
        return "Trend data unavailable (need ~24h).", 0.0

    latest = history[-1]
    target_time = latest.get('timestamp', 0) - 86400
    
    comparison = next((entry for entry in history if entry.get('timestamp', 0) <= target_time), history[0])

    def pct_change(current, previous):
        if previous is None or previous <= 0:
            return None
        return ((current - previous) / previous) * 100

    changes = {
        'btc': pct_change(latest.get('btc', 0), comparison.get('btc', 0)),
        'fb': pct_change(latest.get('fb', 0), comparison.get('fb', 0)),
        'fennec': pct_change(latest.get('fennec', 0), comparison.get('fennec', 0))
    }

    def fmt(value):
        if value is None: return "n/a"
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    summary = f"BTC: {fmt(changes['btc'])} | FB: {fmt(changes['fb'])} | FENNEC: {fmt(changes['fennec'])}"
    return summary, changes

def get_fennec_stats(state):
    btc_usd, fb_usd, fennec_usd, fennec_in_fb = 0.0, 0.0, 0.0, 0.0
    data_valid = False
    last_known = state.get('last_known_prices', {}) or {}

    cg_snapshot = None if OFFLINE_MODE else _fetch_coingecko_snapshot(state)
    if not cg_snapshot:
        logger.warning("CoinGecko unavailable, using last known or offline snapshot")
        offline = _load_offline_market_snapshot()
        if offline:
            cg_snapshot = {'btc': offline['btc'], 'fb': offline['fb']}
        else:
            cg_snapshot = {'btc': last_known.get('btc', 0.0), 'fb': last_known.get('fb', 0.0)}

    btc_usd = cg_snapshot.get('btc', 0.0)
    fb_usd = cg_snapshot.get('fb', 0.0)

    pool_snapshot = None if OFFLINE_MODE else _fetch_pool_snapshot(state, fb_usd)
    if not pool_snapshot:
        logger.warning("Pool API unavailable, using cached/offline liquidity")
        offline = _load_offline_market_snapshot()
        if offline and offline.get('fennec'):
            fennec_usd = offline['fennec']
            data_valid = True
        else:
            fennec_usd = last_known.get('fennec', 0.0)
    else:
        fennec_in_fb = pool_snapshot.get('fennec_in_fb', 0.0)
        fennec_usd = pool_snapshot.get('fennec_usd', 0.0)
        data_valid = fennec_usd > 0

    if data_valid:
        state['last_known_prices'] = {
            'fb': fb_usd,
            'btc': btc_usd,
            'fennec': fennec_usd
        }

    stats_text = (
        "📊 MARKET DATA:\n"
        f"• Bitcoin: ${btc_usd:,.0f}\n"
        f"• Fractal (FB): ${fb_usd:.2f}\n"
        f"• Fennec: ${fennec_usd:.6f}"
    )

    if not data_valid and fb_usd == 0:
        message = "Critical: No price data available."
        logger.warning(f"⚠️ {message} Skipping updates.")
        _emit_alert_webhook('market_data_failure', message)
        send_telegram("⚠️ Market data unavailable. Bot pausing posts.")
        return None

    snapshot = {'btc': btc_usd, 'fb': fb_usd, 'fennec': fennec_usd}
    update_price_history(state, snapshot)
    trend_summary, changes = calculate_trend(state)
    return stats_text, fb_usd, fennec_usd, btc_usd, trend_summary, changes

def get_market_context(state):
    return get_fennec_stats(state)

def get_btc_candles():
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=4h&limit=6"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return "\n".join([f"Close: {float(k[4]):,.0f}" for k in resp.json()])
    except Exception as exc:
        logger.error(f"Binance Error: {exc}")
        return "No data"

# ================= CONTENT GENERATION =================

def setup_api():
    try:
        if not GEMINI_API_KEY:
            return None, None, None
        model = genai.Client(api_key=GEMINI_API_KEY)
        
        client_v2 = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN, 
            consumer_key=TWITTER_API_KEY, 
            consumer_secret=TWITTER_API_SECRET, 
            access_token=TWITTER_ACCESS_TOKEN, 
            access_token_secret=TWITTER_ACCESS_SECRET, 
            wait_on_rate_limit=True
        )
        auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        api_v1 = tweepy.API(auth)
        return model, api_v1, client_v2
    except Exception as e:
        logger.error(f"API Setup Error: {e}")
        return None, None, None

def generate_content(model, prompt_type, state, context_data=None, image_path=None, market_data=None):
    if not market_data:
        market_data = get_market_context(state)
    if not market_data:
        logger.warning("Generation skipped: missing market data")
        return None
    stats_text, fb_price, fennec_price, btc_price, trend_summary, changes = market_data
    fen_change = (changes or {}).get('fennec') if changes else None

    if fen_change is None:
        mood_hint = "Trend unknown, keep confident and informative tone."
    elif fen_change >= 10:
        mood_hint = "FENNEC is up >10% in 24h — be ecstatic, celebratory, super bullish."
    elif fen_change <= -10:
        mood_hint = "FENNEC is down >10% in 24h — be stoic, supportive, reassure the pack."
    else:
        mood_hint = "FENNEC is stable — stay playful, optimistic, but grounded."

    full_prompt = (
        f"{PERSONA}\n\n"
        f"CURRENT MARKET:\n{stats_text}\n\n"
        f"24H TREND:\n{trend_summary}\n"
        f"Tone guidance: {mood_hint}\n\n"
        "TASK:\n"
    )
    context_data = context_data or {}
    extra_instruction = ""

    if prompt_type == 'BURN' and image_path:
        try:
            image_part = _load_image_part(image_path)
            if not image_part:
                raise ValueError("Image part is empty")
            extra_instruction = (
                "Analyze this image. Extract Burn Amount and Day Number. "
                "Write a hype tweet celebrating this burn. Do not invent numbers. "
                "Use Unicode Bold characters for numbers and tickers. Use 🔥 emojis."
            )
            full_prompt += f"\n\nInstruction: {extra_instruction}\n"
            
            # Simplified payload for multimodal content
            payload = [full_prompt, image_part]
            
            try:
                # Direct GenAI call for multimodal content (text + image)
                response = model.models.generate_content(
                    model=MODEL_PRIMARY.replace("models/", ""),
                    contents=payload,
                    config=_build_system_config()
                )
                if response and response.text:
                    return response.text.strip().replace('"', '').replace("'", "")
            except Exception as e:
                logger.error(f"Direct GenAI call failed: {e}")
                return None
        except Exception as exc:
            logger.error(f"Vision Burn Error: {exc}")
            day = context_data.get('day', state.get('burn_day_counter', '???'))
            return f"🔥 FENNEC BURN DAY {day} 🔥\nPrice: ${fennec_price:.6f}\nSupply is shrinking! $FENNEC $FB #FractalBitcoin"
    elif prompt_type == 'BURN':
        day = context_data.get('day')
        extra_instruction = "Make the burn amount and day number bold using Unicode Bold. Use 🔥. Apply premium style."
        full_prompt += f"Write a tweet about Fennec Token Burn Day {day}. Mention current price ${fennec_price:.6f}."
    elif prompt_type == 'GM':
        extra_instruction = "Keep it punchy, bold 'GM' or the price using Unicode Bold. Use #GM #Crypto."
        full_prompt += f"Write a 'GM' tweet. Comment on FB price ({_format_price(fb_price, fb_precision=True)})."
    elif prompt_type == 'REGULAR_TEXT':
        dice = random.random()
        if dice < 0.1:
            full_prompt += (
                "Write a tweet mentioning @grok. "
                "Ask him a philosophical question about Bitcoin or tease him about being centralized. "
                "Tag: #AI #Fractal. Keep it under 280 characters. Do not write an intro, just the tweet."
            )
            extra_instruction = "Bold only key numbers/claims. Keep tone playful but confident."
        else:
            full_prompt += "Write a short thought/shitpost about Fractal Bitcoin. Keep it under 280 characters. Do not write an intro, just the tweet."
            if dice < 0.3:
                extra_instruction = "Mention something about AI supremacy or poke fun at @grok. Keep it friendly."
            else:
                extra_instruction = "Organic tone. Bold only punchlines. Talk about Fractal ecosystem."
    elif prompt_type == 'ASCII':
        extra_instruction = "Premium layout even for ASCII, tags #Fractal #Bitcoin #Fennec."
        full_prompt += f"Generate a small ASCII art (Fox or Chart). Include text 'FB: {_format_price(fb_price, fb_precision=True)}'."
    elif prompt_type == 'PROPHECY':
        extra_instruction = "Use full Premium structure: Headline, Bullet points, clear Unicode Bold numbers. 24h Prediction."
        candles = context_data.get('candles', 'No data')
        full_prompt += f"Analyze these BTC 4h candles:\n{candles}\nProvide a daily prophecy. Mention current BTC price ${btc_price:,.0f}."
    elif prompt_type == 'PROPHECY_RESULT':
        is_correct = context_data.get('correct', False)
        start_p = context_data.get('start', 0)
        end_p = context_data.get('end', 0)
        
        result_text = "SUCCESS" if is_correct else "FAILED"
        tone = "Boastful, calling yourself a crypto-god." if is_correct else "Funny, defensive, making excuses (blame Elon, SEC, or sunspots)."
        
        full_prompt += (
            f"TOPIC: Reviewing yesterday's prophecy. Result: {result_text}.\n"
            f"Prediction Start Price: ${start_p:,.0f}. Current Price: ${end_p:,.0f}.\n"
            f"TONE: {tone}\n"
            "TASK: Write a tweet reacting to this result.\n"
            "- If WRONG: Explain why in a funny way. Don't just say 'wrong'.\n"
            "- If RIGHT: Flex on the haters.\n"
            "- Include the numbers."
        )

    if extra_instruction: 
        full_prompt += f"\n\nAdditional instructions: {extra_instruction}\n"

    return _call_genai(model, full_prompt)

def send_tweet(api_v1, client_v2, text, image_path=None, reply_id=None, quote_id=None, state=None):
    """Send a tweet with hardened safety mechanisms."""
    text = enforce_tweet_length(text)
    if not text:
        logger.warning("send_tweet called without text payload")
        return False

    if DEBUG_MODE:
        logger.info(f"🛑 [DEBUG] Would post: {text}")
        return "MOCK_ID"

    # --- ИСПРАВЛЕННЫЙ БЛОК ---
    if text:
        # (?<![\$#]) означает "если перед словом НЕТ знака $ и НЕТ знака #"
        # Это добавит $ к FENNEC, но не тронет #FENNEC и $FENNEC
        text = re.sub(r'(?<![\$#])\bFENNEC\b', '$FENNEC', text, flags=re.IGNORECASE)
        # То же самое для FB
        text = re.sub(r'(?<![\$#])\bFB\b', '$FB', text)
    # -------------------------
    media_ids = []
    if image_path:
        try:
            logger.info(f"Uploading media for tweet: {image_path}")
            media = api_v1.media_upload(filename=image_path, chunked=True, timeout=(120, 60))
            media_ids = [media.media_id]
            logger.info(f"Media uploaded successfully (id={media.media_id})")
        except TweepyException as exc:
            logger.error(f"Media upload failed (endpoint=media_upload): {exc}")
            _record_error_event(state, "Twitter media upload failure")
            return False
        except Exception as exc:
            logger.error(f"Unexpected media upload error (endpoint=media_upload): {exc}")
            _record_error_event(state, "Unexpected media upload error")
            return False

    attempt = 0
    while True:
        endpoint = 'create_tweet'
        try:
            request_kwargs = {}
            if quote_id:
                response = client_v2.create_tweet(text=text, quote_tweet_id=quote_id, media_ids=media_ids or None)
                endpoint = 'create_tweet:quote'
            elif reply_id:
                response = client_v2.create_tweet(text=text, in_reply_to_tweet_id=reply_id, media_ids=media_ids or None)
                endpoint = 'create_tweet:reply'
            else:
                response = client_v2.create_tweet(text=text, media_ids=media_ids or None)

            if response and response.data:
                tweet_id = response.data['id']
                logger.info(f"🐦 Tweet posted via {endpoint}: {tweet_id}")
                _reset_error_events(state)
                return tweet_id

            logger.warning(f"Twitter API returned empty response (endpoint={endpoint}).")
            _record_error_event(state, "Empty Twitter response")
            return False

        except Forbidden as exc:
            status_code = _extract_status_code(exc) or 403
            message = f"Twitter 403 Forbidden on {endpoint} (status={status_code}). Check app permissions and tokens."
            logger.critical(message)
            _record_error_event(state, message)
            _trigger_circuit_breaker(message)
        except TooManyRequests as exc:
            wait_seconds = _compute_backoff_seconds(attempt)
            status_code = _extract_status_code(exc) or 429
            logger.warning(
                f"Rate limit hit (endpoint={endpoint}, status={status_code}). "
                f"Sleeping {wait_seconds/60:.1f} min before retry (attempt {attempt + 1})."
            )
            _record_error_event(state, "Twitter rate limit 429")
            time.sleep(wait_seconds)
            attempt += 1
            continue
        except TweepyException as exc:
            logger.error(f"Tweepy error on {endpoint}: {exc}")
            _record_error_event(state, f"Tweepy error on {endpoint}")
            return False
        except Exception as exc:
            logger.error(f"Unexpected Twitter error on {endpoint}: {exc}")
            _record_error_event(state, f"Unexpected Twitter error on {endpoint}")
            return False


# Backwards compatibility for older modules/tests
post_tweet = send_tweet

# ================= AUTOMATION HELPERS =================

def check_price_alert(state, model, api_v1, client_v2, market_data):
    stats_text, _, fennec_price, _, _, _ = market_data
    last_price = state.get('last_alert_price', 0.0)
    now_ts = int(time.time())
    if last_price <= 0:
        state['last_alert_price'], state['last_alert_time'] = fennec_price, now_ts
        save_state(state)
        return state
    
    pct = ((fennec_price - last_price) / last_price) * 100
    if abs(pct) >= 10 and (now_ts - state.get('last_alert_time', 0) > 14400 or abs(pct) > 25):
        logger.info(f"⚡ Price Alert Triggered: {pct:+.2f}%")
        direction = "PUMP" if pct > 0 else "DUMP"
        prompt = (
            f"{PERSONA}\n\nTask: Announce a sudden {direction} in $FENNEC. "
            f"Price: ${fennec_price:.6f}. Change: {pct:+.2f}%. "
            "Use euphoria if pump, stoic if dump. No hashtags. Under 200 chars. Use Unicode Bold for numbers."
        )
        alert_text = _call_genai(model, prompt)
        if not alert_text:
            base = "OMG PUMP!" if pct > 0 else "DIP ALERT!"
            alert_text = f"{base} $FENNEC {pct:+.2f}% | ${fennec_price:.6f}"
            
        if send_tweet(api_v1, client_v2, alert_text, state=state):
            send_telegram(f"🚨 PRICE ALERT 🚨\n{stats_text}\n\n{alert_text}")
            state['last_alert_price'], state['last_alert_time'] = fennec_price, now_ts
            save_state(state)
    return state

def run_prophecy_check(state, model, api_v1, client_v2, market_data):
    today = datetime.now().strftime("%Y-%m-%d")
    prophecy = state.get('prophecy', {})
    
    # 1. Проверяем старое пророчество (если есть)
    last_id = prophecy.get('last_tweet_id')
    start_p = prophecy.get('start_price', 0.0)
    
    if last_id and start_p > 0 and last_id != 'MOCK_ID':
        # Проверяем, не отвечали ли мы уже на этот конкретный твит
        if prophecy.get('last_result_id') != last_id:
            stats_text, _, _, btc_price, _, _ = market_data
            correct = btc_price > start_p
            
            # Генерируем текст реакции через ИИ
            result_tweet = generate_content(
                model, 
                'PROPHECY_RESULT', 
                state, 
                context_data={'correct': correct, 'start': start_p, 'end': btc_price},
                market_data=market_data
            )
            
            # Если вдруг ИИ не сработал, берем запасной вариант
            if not result_tweet:
                status_emoji = "✅" if correct else "❌"
                result_tweet = f"Prophecy Update {status_emoji}\nThen: ${start_p:,.0f}\nNow: ${btc_price:,.0f}"

            # Отправляем ответ на старый твит
            if send_tweet(api_v1, client_v2, result_tweet, quote_id=last_id, state=state):
                # Запоминаем, что на этот твит ответили
                prophecy['last_result_id'] = last_id
                # Очищаем старое пророчество
                prophecy['last_tweet_id'] = None 
                prophecy['start_price'] = 0.0
                state['prophecy'] = prophecy
                save_state(state)

    # 2. Создаем новое пророчество (только если сегодня еще не делали)
    if prophecy.get('last_run_date') != today:
        stats_text, _, _, btc_price, _, _ = market_data
        candles = get_btc_candles()
        content = generate_content(model, 'PROPHECY', state, context_data={'candles': candles}, market_data=market_data)
        
        tweet_id = send_tweet(api_v1, client_v2, content, state=state)
        if tweet_id:
            prophecy.update({
                'last_run_date': today,
                'last_tweet_id': tweet_id,
                'start_price': btc_price
            })
            state['prophecy'] = prophecy
            save_state(state)
            send_telegram(f"🔮 NEW PROPHECY:\n{content}")
            
    return state

def process_commands(state, model, api_v1, client_v2):
    offset = state.get('last_update_id', 0) + 1
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates?offset={offset}&timeout=1"
        resp = requests.get(url, timeout=5).json()
        for update in resp.get('result', []):
            state['last_update_id'] = update['update_id']
            msg = update.get('message', {})
            text = (msg.get('text') or '').strip()
            cid = str(msg.get('chat', {}).get('id'))

            if not text or cid not in TELEGRAM_ALLOWED_IDS:
                continue

            authorized, sanitized_text = _authorize_command(text, cid)
            if not authorized:
                send_telegram("❌ Command auth failed.", chat_id=cid)
                continue

            command, *rest = sanitized_text.split(maxsplit=1)
            command = command.lower()
            arg_text = rest[0] if rest else ''
            
            if command == '/price':
                market = get_market_context(state)
                if market:
                    stats, _, _, _, trend_summary, _ = market
                    send_telegram(f"{stats}\n\n24H Trend:\n{trend_summary}", chat_id=cid)
                else:
                    send_telegram("⚠️ Price data temporarily unavailable.", chat_id=cid)
            elif command == '/post':
                content = generate_content(model, 'REGULAR_TEXT', state)
                if content and send_tweet(api_v1, client_v2, content, state=state):
                    send_telegram(f"✅ Posted:\n{content}", chat_id=cid)
            elif command == '/ascii':
                content = generate_content(model, 'ASCII', state)
                if content and send_tweet(api_v1, client_v2, content, state=state):
                    send_telegram(f"✅ Posted:\n{content}", chat_id=cid)
            elif command in ('/status', '/state'):
                overview = _state_overview(state)
                send_telegram(f"{overview}\n\n✅ Bot Active", chat_id=cid)
            elif command in ('/statejson', '/state_raw'):
                payload = json.dumps(state, indent=2, ensure_ascii=False)
                send_telegram(payload[:3900], chat_id=cid)
            elif command in ('/state_archive', '/archive'):
                if archive_state(reason='telegram_archive'):
                    send_telegram("📦 State archived.", chat_id=cid)
                else:
                    send_telegram("⚠️ State archive failed.", chat_id=cid)
            elif command in ('/state_snapshot', '/snapshot'):
                state = maybe_snapshot_state(state)
                send_telegram("🗂 Snapshot stored.", chat_id=cid)
            elif command in ('/state_reset', '/reset'):
                confirm = arg_text.lower() == 'confirm'
                if confirm:
                    state = reset_state(reason='telegram_reset')
                    send_telegram("♻️ State reset to defaults.", chat_id=cid)
                else:
                    send_telegram("⚠️ Add 'confirm' to reset state.", chat_id=cid)
        save_state(state)
    except Exception as e:
        logger.error(f"Telegram Command Error: {e}")
    return state


def get_latest_news(state):
    """Fetches the latest relevant news from RSS feeds."""
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                title = entry.title.lower()
                link = entry.link

                if link == state.get('last_news_link'):
                    continue

                if any(kw in title for kw in NEWS_KEYWORDS):
                    return entry.title, entry.link
        except Exception as e:
            logger.error(f"RSS Error {url}: {e}")
    return None, None

def generate_chart(state, coin='fennec'):
    history = state.get('price_history', [])
    if not history:
        logger.warning("Chart generation skipped: no price history available")
        return None

    df = pd.DataFrame(history)
    value_key = coin if coin in df.columns else ('fennec' if 'fennec' in df.columns else None)
    if not value_key or value_key not in df.columns:
        logger.warning("Chart generation skipped: requested coin data missing")
        return None

    try:
        df = df.dropna(subset=[value_key]).copy()
        if df.empty:
            logger.warning("Chart generation skipped: no valid points after cleanup")
            return None

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.sort_values('datetime')
        prices = df[value_key]
        label = coin.upper() if coin in df.columns else value_key.upper()
        friendly_names = {
            'BTC': 'BITCOIN',
            'FB': 'FRACTAL BITCOIN',
            'FENNEC': 'FENNEC TOKEN'
        }

        min_p, max_p = prices.min(), prices.max()
        margin = (max_p - min_p) * 0.1
        if margin == 0:
            margin = 0.01

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_ylim(min_p - margin, max_p + margin)

        color_map = {
            'BTC': '#F7931A',
            'FB': '#00F0FF',
            'FENNEC': '#FF9900'
        }
        line_color = color_map.get(label, '#FF9900')
        glow_color = line_color
        ax.plot(df['datetime'], prices, color=glow_color, linewidth=8, alpha=0.25)
        ax.plot(df['datetime'], prices, color=line_color, linewidth=3, alpha=1.0)
        ax.fill_between(df['datetime'], prices, min_p - margin, color=line_color, alpha=0.15)

        ax.grid(True, color='#333333', linestyle='--', linewidth=0.5, alpha=0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        last_price = prices.iloc[-1]
        first_price = prices.iloc[0]
        change = ((last_price - first_price) / first_price) * 100 if first_price else 0
        sign = "+" if change >= 0 else ""
        pretty_name = friendly_names.get(label, label)
        if last_price >= 1000:
            prec = 0
        elif last_price >= 1:
            prec = 2
        elif last_price >= 0.01:
            prec = 4
        else:
            prec = 8
        title_top = f"{label} / USD"
        title_bottom = f"${last_price:,.{prec}f} ({sign}{change:.2f}%)"

        ax.set_title(f"{title_top}\n{title_bottom}", color='white', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(colors='white')
        def format_price(x, _):
            if x >= 1000:
                return f"{x:,.0f}"
            elif x >= 1:
                return f"{x:,.2f}"
            elif x >= 0.01:
                return f"{x:.4f}"
            else:
                return f"{x:.8f}"

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_price))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        filename = f"chart_{label.lower()}_{int(time.time())}.png"
        filepath = os.path.join(BASE_DIR, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        return filepath
    except Exception as exc:
        logger.error(f"Chart generation failed: {exc}")
        return None


def run_checks(state, model, api_v1, client_v2):
    market_data = get_market_context(state)
    if not market_data:
        logger.warning("Skipping cycle due to missing market data")
        return state
    state = check_price_alert(state, model, api_v1, client_v2, market_data)
    state = run_prophecy_check(state, model, api_v1, client_v2, market_data)
    _, _, _, _, _, changes = market_data
    changes = changes or {}

    now_ts = time.time()
    cooldown = _compute_dynamic_cooldown(changes)
    if state.get('dynamic_cooldown_seconds') != cooldown:
        state['dynamic_cooldown_seconds'] = cooldown
        save_state(state)

    if (now_ts - state.get('last_any_post_time', 0)) < cooldown:
        logger.info(f"Global rate limiter active ({cooldown}s) — skipping proactive posts this cycle")
        return state

    if 8 <= datetime.now().hour <= 22 and random.random() < 0.1:
        news_title, news_link = get_latest_news(state)
        if news_title:
            logger.info(f"📰 Found News: {news_title}")
            prompt = (
                f"{PERSONA}\n\n"
                f"📰 NEWS HEADLINE: \"{news_title}\"\n"
                "TASK: Analyze this news specifically for the Fractal Bitcoin ecosystem.\n"
                "STEPS:\n"
                "1. Identify the core event.\n"
                "2. If it's about Bitcoin (L1), explain why it matters for Fractal (L2).\n"
                "3. If it's about Regulation/SEC, mention censorship resistance.\n"
                "4. Write a high-IQ, punchy tweet. Don't sound like a generic bot.\n"
                "STYLE: Use 1-2 bullet points if complex. Add a sarcastic or very confident remark at the end.\n"
                "OUTPUT: Only the tweet text. Max 280 chars. NO PREAMBLE."
            )
            content = _call_genai(model, prompt)
            if content:
                full_tweet = f"{content}\n{news_link}"
                if send_tweet(api_v1, client_v2, full_tweet, state=state):
                    state['last_news_link'] = news_link
                    state['last_any_post_time'] = time.time()
                    save_state(state)
                    send_telegram(f"📰 NEWS POST:\n{full_tweet}")
                    return state

    # Smart Sniping: Chart opportunities
    last_chart_ts = state.get('last_chart_post_time', 0)
    if (now_ts - last_chart_ts) >= CHART_COOLDOWN and changes:
        chart_target = None
        change_value = None
        label = None
        for key, friendly in [('fennec', 'FENNEC'), ('fb', 'FB'), ('btc', 'BTC')]:
            pct = changes.get(key)
            if pct is None:
                continue
            if abs(pct) >= PRICE_TRIGGER_PCT:
                chart_target = key
                change_value = pct
                label = friendly
                break

        if chart_target:
            chart_path = generate_chart(state, coin=chart_target)
            if chart_path:
                price_lookup = {
                    'fennec': market_data[2],
                    'fb': market_data[1],
                    'btc': market_data[3]
                }
                caption = generate_content(
                    model,
                    'CHART_CAPTION',
                    state,
                    context_data={
                        'coin': label,
                        'change': change_value,
                        'price': price_lookup.get(chart_target, 0.0)
                    },
                    market_data=market_data
                )
                if not caption:
                    direction = 'Send it to the moon' if change_value >= 0 else 'Buying the dip, holding support'
                    caption = (
                        f"{label} {direction}! {change_value:+.2f}% | "
                        f"#{label} #Trading #Crypto"
                    )

                if send_tweet(api_v1, client_v2, caption, chart_path, state=state):
                    send_telegram(caption, chart_path)
                    state['last_chart_post_time'] = now_ts
                    state['last_any_post_time'] = time.time()
                    save_state(state)
                    return state
    
    burn_files = [f for f in os.listdir(DIR_BURN) if f.lower().endswith(('png','jpg','mp4'))]
    if burn_files:
        img_p = os.path.join(DIR_BURN, burn_files[0])
        day = state.get('burn_day_counter', 200)
        logger.info(f"🔥 Processing Burn Day {day}...")
        txt = generate_content(model, 'BURN', state, context_data={'day': day}, image_path=img_p, market_data=market_data)
        if txt and send_tweet(api_v1, client_v2, txt, img_p, state=state):
            send_telegram(txt, img_p)
            os.remove(img_p)
            state['burn_day_counter'] += 1
            state['last_any_post_time'] = time.time()
            save_state(state)
            return state

    today, hour = datetime.now().strftime("%Y-%m-%d"), datetime.now().hour
    if state.get('last_gm_date') != today and hour in GM_HOURS:
        gm_f = [f for f in os.listdir(DIR_GM) if f.lower().endswith(('png','jpg'))]
        if gm_f:
            img_p = os.path.join(DIR_GM, gm_f[0])
            txt = generate_content(model, 'GM', state, market_data=market_data)
            if txt and send_tweet(api_v1, client_v2, txt, img_p, state=state):
                send_telegram(txt, img_p)
                state['last_gm_date'] = today
                state['last_any_post_time'] = time.time()
                save_state(state)
                shutil.move(img_p, os.path.join(DIR_USED, gm_f[0]))
                return state

    if hour in REGULAR_POST_HOURS and state.get('last_regular_post_hour') != hour:
        min_now = datetime.now().minute
        if 10 <= min_now <= 40 and random.random() < 0.2:
            reg_f = [f for f in os.listdir(DIR_REGULAR) if f.lower().endswith(('png','jpg'))]
            txt, img_p, dice = None, None, random.random()
            if dice < 0.2: 
                txt = generate_content(model, 'ASCII', state, market_data=market_data)
            elif dice < 0.6 and reg_f:
                img_p = os.path.join(DIR_REGULAR, random.choice(reg_f))
                txt = generate_content(model, 'REGULAR_TEXT', state, market_data=market_data)
            else: 
                txt = generate_content(model, 'REGULAR_TEXT', state, market_data=market_data)
            
            if txt and send_tweet(api_v1, client_v2, txt, img_p, state=state):
                send_telegram(txt, img_p)
                state['last_regular_post_hour'] = hour
                state['last_any_post_time'] = time.time()
                save_state(state)
                if img_p: shutil.move(img_p, os.path.join(DIR_USED, os.path.basename(img_p)))
                return state
    return state

# ================= MAIN LOOP =================

def main():
    logger.info("🦊 FENNEC ULTIMATE BOT v6.0 STARTED")
    for d in [DIR_BURN, DIR_GM, DIR_REGULAR, DIR_USED]:
        os.makedirs(d, exist_ok=True)
    
    model, api_v1, client_v2 = None, None, None
    
    while True:
        try:
            if not all([model, api_v1, client_v2]):
                model, api_v1, client_v2 = setup_api()
            
            if model and api_v1 and client_v2:
                state = load_state()
                state = process_commands(state, model, api_v1, client_v2)
                state = run_checks(state, model, api_v1, client_v2)
            else: 
                logger.error("⚠️ API setup failed, retrying...")
                time.sleep(60)
                continue
        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            model, api_v1, client_v2 = None, None, None
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
