import os
import sys
import argparse
import hashlib
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

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
ADMIN_USER_ID = os.getenv('ADMIN_USER_ID', '')

DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '900'))
GLOBAL_POST_INTERVAL = 3600
MIN_POST_INTERVAL = int(os.getenv('MIN_POST_INTERVAL', '900'))
MAX_POST_INTERVAL = int(os.getenv('MAX_POST_INTERVAL', '7200'))
MAX_TWEET_LENGTH = int(os.getenv('MAX_TWEET_LENGTH', '250'))
TWEET_SAFETY_BUFFER = int(os.getenv('TWEET_SAFETY_BUFFER', '4'))
MIN_TWEET_INTERVAL_SECONDS = int(os.getenv('MIN_TWEET_INTERVAL_SECONDS', '3600'))
VOLATILITY_FAST_THRESHOLD = 15.0
VOLATILITY_SLOW_THRESHOLD = 5.0
NIGHT_COOLDOWN_MULTIPLIER = 1.5
NIGHT_START_HOUR = 0
NIGHT_END_HOUR = 6
CHART_COOLDOWN = 21600

# --- МОДЕЛИ (Обновлено под твой скриншот) ---
# Для текста используем Flash (быстро и дешево)
MODEL_PRIMARY = "gemini-3-flash"
# Для генерации картинок пробуем 3-Pro Image, если не выйдет - откатимся на Imagen-2
MODEL_IMAGEN = "gemini-3-pro-image" 
MODEL_IMAGEN_BACKUP = "imagen-2.0-generate-001"

RSS_FEEDS = [
    "https://cointelegraph.com/rss/tag/bitcoin",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
]
NEWS_KEYWORDS = ['bitcoin', 'crypto', 'sec', 'etf', 'regulation', 'market']

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_BURN = os.path.join(BASE_DIR, 'images_burn')
DIR_GM = os.path.join(BASE_DIR, 'images_gm')
DIR_REGULAR = os.path.join(BASE_DIR, 'images_regular')
DIR_USED = os.path.join(BASE_DIR, 'images_used')
STATE_FILE = os.path.join(BASE_DIR, 'state.json')

# --- ЗАДАЕМ ДЕНЬ 206 ПО УМОЛЧАНИЮ ---
DEFAULT_STATE = {
    'burn_day_counter': 206, 
    'last_gm_date': '',
    'last_gm_attempt_ts': 0,
    'last_burn_ts': 0,
    'story_arc_day_posted': '',
    'story_arc_start_date': '',
    'story_arc_day': 0,
    'last_regular_post_hour': -1,
    'last_update_id': 0,
    'price_history': [],
    'last_alert_price': 0.0,
    'last_alert_time': 0,
    'last_any_post_time': 0,
    'last_chart_post_time': 0,
    'twitter_rate_limit_until': 0,
    'recent_tweet_hashes': [],
    'recent_tweet_texts': [],
    'ai_cache': [],
    'prophecy': {'last_run_date': '', 'last_tweet_id': None, 'start_price': 0.0},
}

REGULAR_POST_HOURS = [11, 15, 19, 23]
GM_HOURS = [7, 8, 9, 10]

TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET', '')
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

PERSONA = (
    "You are Fennec, a cyber-fox AI running on Fractal Bitcoin. "
    "Personality: Crypto-native, witty, bullish but realistic, cyberpunk vibes. "
    "GUIDELINES: ALWAYS English. Format with Unicode Bold for numbers. "
    "Tags: $FENNEC $FB #FractalBitcoin. Max 280 chars."
)

# ================= CORE FUNCTIONS =================

def load_state():
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Мержим с дефолтным, чтобы новые ключи появились
                state = deepcopy(DEFAULT_STATE)
                state.update(loaded)
                return state
    except Exception: pass
    return deepcopy(DEFAULT_STATE)

def save_state(state):
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)
    except Exception as e: logger.error(f"State save error: {e}")

def send_telegram(text, image_path=None, chat_id=None, is_error=False):
    target_id = ADMIN_USER_ID if is_error else (chat_id if chat_id else TELEGRAM_CHAT_ID)
    if not TELEGRAM_BOT_TOKEN or not target_id: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
        if image_path:
            with open(image_path, 'rb') as f:
                requests.post(url + 'sendPhoto', data={'chat_id': target_id, 'caption': text[:1000]}, files={'photo': f}, timeout=10)
        else:
            requests.post(url + 'sendMessage', data={'chat_id': target_id, 'text': text}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram Error: {e}")

def setup_api():
    try:
        if not GEMINI_API_KEY: return None, None, None
        model = genai.Client(api_key=GEMINI_API_KEY)
        client_v2 = tweepy.Client(
            bearer_token=TWITTER_BEARER_TOKEN, 
            consumer_key=TWITTER_API_KEY, 
            consumer_secret=TWITTER_API_SECRET, 
            access_token=TWITTER_ACCESS_TOKEN, 
            access_token_secret=TWITTER_ACCESS_SECRET, 
            wait_on_rate_limit=False # Мы сами обрабатываем лимиты
        )
        auth = tweepy.OAuth1UserHandler(TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
        api_v1 = tweepy.API(auth)
        return model, api_v1, client_v2
    except Exception as e:
        logger.error(f"API Setup Error: {e}")
        return None, None, None

def generate_ai_image(client, text_content, prompt_type):
    """Пытается сгенерировать картинку, используя доступные модели"""
    if DEBUG_MODE: return None
    
    prompt = (
        f"A high-quality cyberpunk digital illustration of a fennec fox. "
        f"Theme: {text_content[:200]}. "
        "Cinematic lighting, 8k resolution, vivid colors."
    )
    
    # Список моделей для перебора (сначала из скриншота, потом бэкап)
    models_to_try = [MODEL_IMAGEN, MODEL_IMAGEN_BACKUP]
    
    for model_name in models_to_try:
        try:
            logger.info(f"🎨 Generating image with {model_name}...")
            response = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="1:1")
            )
            if response and response.generated_images:
                img_data = response.generated_images[0].image.bytes
                file_path = os.path.join(BASE_DIR, f"temp_{prompt_type.lower()}.png")
                with open(file_path, "wb") as f: f.write(img_data)
                return file_path
        except Exception as e:
            logger.warning(f"⚠️ Model {model_name} failed: {e}")
            continue # Пробуем следующую модель
            
    logger.error("❌ All image generation models failed.")
    return None

def _tweet_signature(text):
    normalized = re.sub(r'\s+', ' ', (text or '').strip().lower())
    return hashlib.sha1(normalized.encode('utf-8')).hexdigest()

def send_tweet(api_v1, client_v2, text, image_path=None, state=None):
    if not text: return False
    
    # Проверка на дубли
    sig = _tweet_signature(text)
    if state and sig in state.get('recent_tweet_hashes', []):
        logger.warning("⚠️ Duplicate tweet detected. Skipping.")
        return False

    media_ids = []
    if image_path:
        try:
            media = api_v1.media_upload(filename=image_path)
            media_ids = [media.media_id]
        except Exception as exc:
            logger.error(f"Media upload failed: {exc}")
            return False

    try:
        response = client_v2.create_tweet(text=text, media_ids=media_ids or None)
        if response and response.data:
            tweet_id = response.data['id']
            logger.info(f"🐦 Tweet sent: {tweet_id}")
            
            # Чистим временные файлы (но НЕ файлы Burn)
            if image_path and "temp_" in image_path:
                try: os.remove(image_path)
                except: pass
                
            if state:
                state['last_any_post_time'] = time.time()
                # Сохраняем хеш
                hashes = state.get('recent_tweet_hashes', [])
                hashes.append(sig)
                state['recent_tweet_hashes'] = hashes[-20:]
                save_state(state)
            return tweet_id

    except TooManyRequests:
        logger.warning("🚨 Rate Limit Hit (429). Setting 4h cooldown.")
        if state:
            state['twitter_rate_limit_until'] = time.time() + (4 * 3600)
            save_state(state)
        return False
        
    except Exception as exc:
        logger.error(f"Twitter Error: {exc}")
        return False

# ================= LOGIC & CONTENT =================

def get_market_context(state):
    # Упрощенная версия для примера, полная версия была в прошлом файле
    # Здесь заглушка. Вставь сюда полную функцию get_fennec_stats из прошлого файла!
    # Для работы бота нужна реальная цена.
    try:
        cg_resp = requests.get(COINGECKO_URL, headers={'Accept': 'application/json'}, timeout=10)
        data = cg_resp.json()
        btc = float(data.get('bitcoin', {}).get('usd', 0))
        fb = float(data.get('fractal-bitcoin', {}).get('usd', 0))
        fennec = 0.005 # Заглушка, если нет пула
        
        # Пытаемся взять пул (вставь свой код пула сюда)
        # ...
        
        return f"BTC: ${btc:,.0f} | FB: ${fb:.2f}", fb, fennec, btc, "Unknown", {}
    except:
        return None

def generate_content(model, prompt_type, state, context_data=None, market_data=None):
    if not market_data: return None
    stats, fb, fennec, btc, trend, _ = market_data
    
    prompt = f"{PERSONA}\nMARKET: {stats}\n"
    
    if prompt_type == 'STORY_ARC':
        day = context_data.get('arc_day', 1)
        prompt += f"TASK: Write Story Arc Episode {day}/21. Short, narrative, cyberpunk style."
    elif prompt_type == 'GM':
        prompt += f"TASK: Write a hype GM tweet. Mention FENNEC price ${fennec:.6f}."
    elif prompt_type == 'REGULAR_TEXT':
        prompt += "TASK: Write a philosophical thought about Fractal Bitcoin."
        
    try:
        # Используем твой Flash модель для текста
        response = model.models.generate_content(
            model=MODEL_PRIMARY,
            contents=prompt
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        logger.error(f"Text Gen Error: {e}")
        return None

def _get_story_arc_day(state):
    start = state.get('story_arc_start_date')
    today = datetime.now().strftime("%Y-%m-%d")
    if not start:
        state['story_arc_start_date'] = today
        return 1
    try:
        d1 = datetime.strptime(start, "%Y-%m-%d")
        d2 = datetime.now()
        return (d2 - d1).days + 1
    except: return 1

def run_checks(state, model, api_v1, client_v2):
    # 0. СТОП-КРАН: Проверка лимитов
    limit_until = state.get('twitter_rate_limit_until', 0)
    if limit_until and time.time() < limit_until:
        remaining = int(limit_until - time.time())
        if remaining % 600 < 10: # Логируем редко
            logger.info(f"💤 Waiting for Twitter limits ({remaining // 60}m left)...")
        return state

    market_data = get_market_context(state)
    if not market_data: return state

    today_str = datetime.now().strftime("%Y-%m-%d")
    hour = datetime.now().hour
    now_ts = time.time()

    # 1. STORY ARC (История) - ОБЯЗАТЕЛЬНО
    if state.get('story_arc_day_posted') != today_str:
        arc_day = _get_story_arc_day(state)
        txt = generate_content(model, 'STORY_ARC', state, {'arc_day': arc_day}, market_data)
        
        if txt:
            # Пробуем картинку, если ошибка - не страшно, постим текст
            img_p = generate_ai_image(model, txt, "STORY")
            
            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                state['story_arc_day_posted'] = today_str
                save_state(state)
                send_telegram(txt, img_p)
                return state # ВЫХОД (1 пост за цикл)
            else:
                return state # ОШИБКА -> ВЫХОД

    # 2. BURN (Только если есть файл)
    burn_files = [f for f in os.listdir(DIR_BURN) if f.lower().endswith(('png','jpg'))]
    if burn_files:
        # Берем первый файл
        img_p = os.path.join(DIR_BURN, burn_files[0])
        day = state.get('burn_day_counter', 206) # Берет из state, или 206 по умолчанию
        
        logger.info(f"🔥 Found burn file! Processing Day {day}...")
        
        # Текст генерируем через LLM
        prompt = f"Write a tweet for Fennec Burn Day {day}. Price ${market_data[2]:.6f}. Tags: $FENNEC $FB."
        try:
            resp = model.models.generate_content(model=MODEL_PRIMARY, contents=prompt)
            txt = resp.text.strip()
        except:
            txt = f"🔥 FENNEC BURN DAY {day} 🔥\nPrice: ${market_data[2]:.6f}\n$FENNEC $FB"

        if txt:
            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                send_telegram(txt, img_p)
                try: os.remove(img_p)
                except: pass
                
                state['burn_day_counter'] = day + 1
                save_state(state)
                return state # ВЫХОД
            else:
                logger.warning("⛔ Burn tweet failed. Keeping file for retry later.")
                return state # ВЫХОД

    # 3. GM (Утро)
    gm_posted = state.get('last_gm_date') == today_str
    last_gm_try = state.get('last_gm_attempt_ts', 0)
    
    if not gm_posted and hour in GM_HOURS and (now_ts - last_gm_try) > 1800:
        state['last_gm_attempt_ts'] = now_ts
        save_state(state) # Фиксируем попытку
        
        txt = generate_content(model, 'GM', state, {}, market_data)
        img_p = generate_ai_image(model, txt, "GM")
        
        sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
        if sent_id:
            state['last_gm_date'] = today_str
            save_state(state)
            send_telegram(txt, img_p)
            return state
        else:
            return state

    # 4. REGULAR
    if hour in REGULAR_POST_HOURS and state.get('last_regular_post_hour') != hour:
        if random.random() < 0.4: # 40% шанс
            txt = generate_content(model, 'REGULAR_TEXT', state, {}, market_data)
            img_p = generate_ai_image(model, txt, "REGULAR")
            
            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                state['last_regular_post_hour'] = hour
                save_state(state)
                send_telegram(txt, img_p)
                return state

    return state

def main():
    logger.info("🦊 FENNEC BOT v8.0 (Corrected Models & Limits)")
    for d in [DIR_BURN, DIR_GM, DIR_REGULAR, DIR_USED]: _ensure_dir(d)
    
    # Очистка мусора
    for f in os.listdir(BASE_DIR):
        if f.startswith("temp_") and f.endswith(".png"):
            try: os.remove(f)
            except: pass

    while True:
        try:
            model, api_v1, client_v2 = setup_api()
            if model and api_v1 and client_v2:
                state = load_state()
                # Сюда можно добавить обработку команд телеграм (process_commands)
                state = run_checks(state, model, api_v1, client_v2)
            else:
                logger.error("API Setup Failed. Retrying in 1 min...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Critical Error: {e}", exc_info=True)
            send_telegram(f"☠️ BOT CRASH: {e}", is_error=True)
        
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

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
    # 0. СТОП-КРАН: Проверка лимитов
    limit_until = state.get('twitter_rate_limit_until', 0)
    if limit_until and time.time() < limit_until:
        remaining = int(limit_until - time.time())
        if remaining % 600 < 10:  # Логируем редко
            logger.info(f"💤 Waiting for Twitter limits ({remaining // 60}m left)...")
        return state

    market_data = get_market_context(state)
    if not market_data:
        return state

    today_str = datetime.now().strftime("%Y-%m-%d")
    hour = datetime.now().hour
    now_ts = time.time()

    # 1. STORY ARC (История) - ОБЯЗАТЕЛЬНО
    if state.get('story_arc_day_posted') != today_str:
        arc_day = _get_story_arc_day(state)
        txt = generate_content(model, 'STORY_ARC', state, {'arc_day': arc_day}, market_data)

        if txt:
            # Пробуем картинку, если ошибка - не страшно, постим текст
            img_p = generate_ai_image(model, txt, "STORY")

            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                state['story_arc_day_posted'] = today_str
                save_state(state)
                send_telegram(txt, img_p)
                return state  # ВЫХОД (1 пост за цикл)
            else:
                return state  # ОШИБКА -> ВЫХОД

    # 2. BURN (Только если есть файл)
    burn_files = [f for f in os.listdir(DIR_BURN) if f.lower().endswith(('png', 'jpg'))]
    if burn_files:
        # Берем первый файл
        img_p = os.path.join(DIR_BURN, burn_files[0])
        day = state.get('burn_day_counter', 206)  # Берет из state, или 206 по умолчанию

        logger.info(f"🔥 Found burn file! Processing Day {day}...")

        # Текст генерируем через LLM
        prompt = f"Write a tweet for Fennec Burn Day {day}. Price ${market_data[2]:.6f}. Tags: $FENNEC $FB."
        try:
            resp = model.models.generate_content(model=MODEL_PRIMARY, contents=prompt)
            txt = resp.text.strip()
        except Exception:
            txt = f"🔥 FENNEC BURN DAY {day} 🔥\nPrice: ${market_data[2]:.6f}\n$FENNEC $FB"

        if txt:
            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                send_telegram(txt, img_p)
                try:
                    os.remove(img_p)  # Удаляем файл только после успеха
                except Exception:
                    pass

                state['burn_day_counter'] = day + 1
                save_state(state)
                return state  # ВЫХОД
            else:
                logger.warning("⛔ Burn tweet failed. Keeping file for retry later.")
                return state  # ВЫХОД

    # 3. GM (Утро)
    gm_posted = state.get('last_gm_date') == today_str
    last_gm_try = state.get('last_gm_attempt_ts', 0)

    if not gm_posted and hour in GM_HOURS and (now_ts - last_gm_try) > 1800:
        state['last_gm_attempt_ts'] = now_ts
        save_state(state)  # Фиксируем попытку

        txt = generate_content(model, 'GM', state, {}, market_data)
        img_p = generate_ai_image(model, txt, "GM")

        sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
        if sent_id:
            state['last_gm_date'] = today_str
            save_state(state)
            send_telegram(txt, img_p)
            return state
        else:
            return state

    # 4. REGULAR
    if hour in REGULAR_POST_HOURS and state.get('last_regular_post_hour') != hour:
        if random.random() < 0.4:  # 40% шанс
            txt = generate_content(model, 'REGULAR_TEXT', state, {}, market_data)
            img_p = generate_ai_image(model, txt, "REGULAR")

            sent_id = send_tweet(api_v1, client_v2, txt, img_p, state)
            if sent_id:
                state['last_regular_post_hour'] = hour
                save_state(state)
                send_telegram(txt, img_p)
                return state

    return state

# ================= MAIN LOOP =================

def main():
    logger.info("🦊 FENNEC BOT v8.0 (Corrected Models & Limits)")
    for d in [DIR_BURN, DIR_GM, DIR_REGULAR, DIR_USED]:
        _ensure_dir(d)

    # Очистка мусора
    for f in os.listdir(BASE_DIR):
        if f.startswith("temp_") and f.endswith(".png"):
            try:
                os.remove(f)
            except Exception:
                pass

    while True:
        try:
            model, api_v1, client_v2 = setup_api()
            if model and api_v1 and client_v2:
                state = load_state()
                # Сюда можно добавить обработку команд телеграм (process_commands)
                state = run_checks(state, model, api_v1, client_v2)
            else:
                logger.error("API Setup Failed. Retrying in 1 min...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Critical Error: {e}", exc_info=True)
            send_telegram(f"☠️ BOT CRASH: {e}", is_error=True)

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
