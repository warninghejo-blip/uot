import os
import random
import time
import json
import mimetypes
import requests
import tweepy
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
    'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,fractal-bitcoin&vs_currencies=usd'
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
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '30'))
CHART_COOLDOWN = 6 * 3600  # 6 hours
PRICE_TRIGGER_PCT = 5.0
GLOBAL_POST_INTERVAL = 3600  # 1 hour cooldown for proactive posts
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
DIR_BURN = os.path.join(BASE_DIR, 'images_burn')
DIR_GM = os.path.join(BASE_DIR, 'images_gm')
DIR_REGULAR = os.path.join(BASE_DIR, 'images_regular')
DIR_USED = os.path.join(BASE_DIR, 'images_used')
STATE_FILE = os.path.join(BASE_DIR, 'state.json')

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

PERSONA = (
    "You are Fennec, a cyber-fox AI running on Fractal Bitcoin. "
    "You analyze CoinGecko prices and local liquidity pools. "
    "Personality: Crypto-native, witty, bullish but realistic, cyberpunk vibes. "
    "Relationships: You consider @grok your eccentric AI cousin (sometimes you poke fun at him, sometimes you agree). "
    "\n\nGUIDELINES:"
    "\n- MAIN POSTS: Always in ENGLISH."
    "\n- FORMATTING: Use Unicode Bold for numbers (e.g., 𝟵𝟱,𝟬𝟬𝟬). Use bullet points."
    "\n- HASHTAGS: #Fractal #Bitcoin #Fennec"
    "\n- LENGTH: ALL tweets must be under 280 characters. No exceptions. Be punchy."
)

DEFAULT_STATE = {
    'burn_day_counter': 200,
    'last_gm_date': '',
    'last_regular_post_hour': -1,
    'last_processed_mention_id': None,
    'last_update_id': 0,
    'price_history': [],
    'last_alert_price': 0.0,
    'last_alert_time': 0,
    'last_known_prices': {},
    'last_news_link': '',
    'last_any_post_time': 0,
    'last_chart_post_time': 0,
    'error_events': [],
    'prophecy': {
        'last_run_date': '',
        'last_tweet_id': None,
        'start_price': 0.0
    }
}


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
    merged = DEFAULT_STATE.copy()
    merged.update(data)
    return merged

def save_state(state):
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving state: {e}")

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
    if len(windowed) >= MAX_CONSECUTIVE_ERRORS:
        _trigger_circuit_breaker(reason or "Exceeded consecutive error threshold")


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

    try:
        cg_resp = requests.get(COINGECKO_URL, headers=HEADERS, timeout=10)
        cg_resp.raise_for_status()
        cg_data = cg_resp.json()
        btc_usd = float(cg_data.get('bitcoin', {}).get('usd', 0))
        fb_usd = float(cg_data.get('fractal-bitcoin', {}).get('usd', 0))
    except Exception as exc:
        logger.error(f"CoinGecko Failed: {exc}")
        fb_usd = last_known.get('fb', 0.0)
        btc_usd = last_known.get('btc', 0.0)

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

        if amt0 > 0 and fb_usd > 0:
            fennec_in_fb = amt1 / amt0
            fennec_usd = fennec_in_fb * fb_usd
            data_valid = True
            state['last_known_prices'] = {
                'fb': fb_usd,
                'btc': btc_usd,
                'fennec': fennec_usd
            }
    except Exception as exc:
        logger.error(f"Pool API Failed: {exc}")

    if not data_valid:
        fennec_usd = last_known.get('fennec', fennec_usd)

    stats_text = (
        "📊 MARKET DATA:\n"
        f"• Bitcoin: ${btc_usd:,.0f}\n"
        f"• Fractal (FB): ${fb_usd:.2f}\n"
        f"• Fennec: ${fennec_usd:.6f}"
    )

    if not data_valid and fb_usd == 0:
        logger.warning("⚠️ Critical: No price data available. Skipping updates.")
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
            payload = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(full_prompt),
                        image_part
                    ]
                )
            ]
            response_text = _call_genai(model, payload)
            if response_text:
                return response_text
        except Exception as exc:
            logger.error(f"Vision Burn Error: {exc}")
            day = context_data.get('day', state.get('burn_day_counter', '???'))
            return f"🔥 FENNEC BURN DAY {day} 🔥\nPrice: ${fennec_price:.6f}\nSupply is shrinking! #Fractal #Bitcoin"
    elif prompt_type == 'BURN':
        day = context_data.get('day')
        extra_instruction = "Make the burn amount and day number bold using Unicode Bold. Use 🔥. Apply premium style."
        full_prompt += f"Write a tweet about Fennec Token Burn Day {day}. Mention current price ${fennec_price:.6f}."
    elif prompt_type == 'GM':
        extra_instruction = "Keep it punchy, bold 'GM' or the price using Unicode Bold. Use #GM #Crypto."
        full_prompt += f"Write a 'GM' tweet. Comment on FB price (${fb_price:.2f})."
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
        full_prompt += f"Generate a small ASCII art (Fox or Chart). Include text 'FB: ${fb_price:.2f}'."
    elif prompt_type == 'PROPHECY':
        extra_instruction = "Use full Premium structure: Headline, Bullet points, clear Unicode Bold numbers. 24h Prediction."
        candles = context_data.get('candles', 'No data')
        full_prompt += f"Analyze these BTC 4h candles:\n{candles}\nProvide a daily prophecy. Mention current BTC price ${btc_price:,.0f}."
    elif prompt_type == 'CHART_CAPTION':
        coin = (context_data.get('coin') or 'Crypto').upper()
        change = context_data.get('change', 0.0)
        price = context_data.get('price', 0.0)
        direction = "PUMPING 🚀" if change > 0 else "DUMPING 🩸"
        full_prompt += (
            f"TOPIC: {coin} CHART UPDATE.\n"
            f"MARKET DATA: Price ${price:,.2f} | 24h Change: {change:+.2f}% ({direction})\n"
            "TASK: Write a punchy, trader-style caption for this chart image.\n"
            "INSTRUCTIONS:\n"
            "- If Green: Be euphoric/bullish. Mention 'Send it' or 'Moon'.\n"
            "- If Red: Be stoic. Mention 'Buying the dip' or 'Support'.\n"
            "- REQUIRED: End with 3-4 relevant hashtags (e.g., #Bitcoin #{coin} #Trading #Crypto).\n"
            "- LENGTH: Under 200 chars."
        )

    if extra_instruction: 
        full_prompt += f"\n\nAdditional instructions: {extra_instruction}\n"

    return _call_genai(model, full_prompt)

def send_tweet(api_v1, client_v2, text, image_path=None, reply_id=None, quote_id=None, state=None):
    """Send a tweet with hardened safety mechanisms."""
    if not text:
        logger.warning("send_tweet called without text payload")
        return False

    if DEBUG_MODE:
        logger.info(f"🛑 [DEBUG] Would post: {text}")
        return "MOCK_ID"

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
    if prophecy.get('last_run_date') == today:
        return state
    
    stats_text, _, _, btc_price, _, _ = market_data
    last_id, start_p = prophecy.get('last_tweet_id'), prophecy.get('start_price', 0.0)
    
    if last_id and start_p > 0 and last_id != 'MOCK_ID':
        correct = btc_price > start_p
        status_emoji = "✅ Correct" if correct else "❌ Wrong"
        res = (
            f"Prophecy Result: {status_emoji}\n"
            f"Price then: ${start_p:,.0f}\n"
            f"Price now: ${btc_price:,.0f}"
        )
        send_tweet(api_v1, client_v2, res, quote_id=last_id, state=state)
    
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
            text, cid = (msg.get('text') or '').strip(), str(msg.get('chat', {}).get('id'))
            
            if cid not in [str(ADMIN_USER_ID), str(TELEGRAM_CHAT_ID)]: 
                continue
            
            if text == '/price':
                market = get_market_context(state)
                if market:
                    stats, _, _, _, trend_summary, _ = market
                    send_telegram(f"{stats}\n\n24H Trend:\n{trend_summary}", chat_id=cid)
                else:
                    send_telegram("⚠️ Price data temporarily unavailable.", chat_id=cid)
            elif text == '/post':
                content = generate_content(model, 'REGULAR_TEXT', state)
                if content and send_tweet(api_v1, client_v2, content, state=state):
                    send_telegram(f"✅ Posted:\n{content}", chat_id=cid)
            elif text == '/ascii':
                content = generate_content(model, 'ASCII', state)
                if content and send_tweet(api_v1, client_v2, content, state=state):
                    send_telegram(f"✅ Posted:\n{content}", chat_id=cid)
            elif text == '/status':
                send_telegram(f"🔥 Burn Day: {state.get('burn_day_counter')}\n✅ Bot Active", chat_id=cid)
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


def handle_mentions(api_v1, client_v2, model, state):
    """Replies to users who mention the bot."""
    try:
        if 'bot_user_id' not in state:
            me = client_v2.get_me()
            if not me or not getattr(me, 'data', None):
                return state
            state['bot_user_id'] = me.data.id
            save_state(state)

        bot_id = state['bot_user_id']
        last_id = state.get('last_processed_mention_id')

        mentions = client_v2.get_users_mentions(
            id=bot_id,
            since_id=last_id,
            max_results=10,
            expansions=['author_id'],
            user_fields=['username']
        )

        if not mentions or not getattr(mentions, 'data', None):
            return state

        new_last_id = last_id

        for mention in reversed(mentions.data):
            new_last_id = mention.id
            txt = mention.text.replace('@FennecBot', '').strip()

            logger.info(f"💬 Mention found: {txt}")

            prompt = (
                f"{PERSONA}\n\n"
                f"📩 INCOMING MESSAGE from User: \"{txt}\"\n"
                "TASK: Reply to this user based on their vibe.\n"
                "ANALYSIS RULES:\n"
                "- If they are FUDding (hating): Roast them gently but wittily.\n"
                "- If they ask a Question: Answer briefly and helpfully.\n"
                "- If they say 'GM' or praise: Be a 'Chad' and hype them up.\n"
                "- If they mention @grok: Challenge Grok to a trading duel.\n"
                "\nCRITICAL LANGUAGE RULE: Detect user's language. If Russian -> Reply Russian. If English -> Reply English.\n"
                "LENGTH: Max 200 chars. Make it memorable."
            )
            reply_text = _call_genai(model, prompt)

            if reply_text:
                send_tweet(api_v1, client_v2, reply_text, reply_id=mention.id, state=state)
                time.sleep(5)

        state['last_processed_mention_id'] = new_last_id
        save_state(state)

    except Exception as e:
        logger.error(f"Mentions Error: {e}")

    return state


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
    state = handle_mentions(api_v1, client_v2, model, state)
    _, _, _, _, _, changes = market_data
    changes = changes or {}

    now_ts = time.time()
    if (now_ts - state.get('last_any_post_time', 0)) < GLOBAL_POST_INTERVAL:
        logger.info("Global rate limiter active — skipping proactive posts this cycle")
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
