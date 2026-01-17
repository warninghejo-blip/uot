import os
import time
import logging
from dotenv import load_dotenv
from google import genai
import requests
import bot

logging.basicConfig(level=logging.INFO, format='%(message)s')


def open_file(path):
    try:
        os.startfile(path)
        print(f"🖼️ Opening image: {path}")
    except Exception:
        print(f"✅ Image saved at: {path} (open manually)")


def get_real_coingecko_chart():
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1"
        data = requests.get(url, timeout=10).json()
        prices_btc = data.get('prices', [])
        if not prices_btc:
            raise ValueError("Empty price series from CoinGecko")

        history = []
        for ts_ms, btc_price in prices_btc:
            history.append({
                'timestamp': ts_ms / 1000,
                'fennec': btc_price,  # placeholder but keeps structure
                'fb': btc_price,
                'btc': btc_price
            })
        return history
    except Exception as exc:
        logging.error(f"Failed to fetch CoinGecko data: {exc}")
        return []


def run_visual_simulation():
    print("\n🎨 STARTING VISUAL SIMULATION (Charts & Charts)...")

    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    client = genai.Client(api_key=api_key) if api_key else None

    def print_tweet_mock(text, image_path=None, **kwargs):
        print(f"\n📢 [TWEET CONTENT]:\n{'-'*40}\n{text}\n{'-'*40}")
        if image_path:
            print(f"📎 ATTACHMENT: {image_path}")
            open_file(image_path)
        else:
            print("📎 NO IMAGE")
        return "MOCK_ID"

    bot.post_tweet = print_tweet_mock

    print("\n📊 Fetching 24H CoinGecko history...")
    state = bot.DEFAULT_STATE.copy()
    history = get_real_coingecko_chart()
    if not history:
        print("❌ Could not fetch CoinGecko history. Aborting simulation.")
        return
    state['price_history'] = history
    bot.save_state(state)

    print("\n\n🧪 TEST 1: CHART GENERATION")
    if hasattr(bot, 'generate_chart'):
        chart_path = bot.generate_chart(state, coin='btc')
        if chart_path:
            print("✅ Chart generated successfully.")
            btc_price = history[-1]['btc'] if history else 0
            stats = f"Bitcoin: ${btc_price:,.0f}"
            market_data = (stats, btc_price, btc_price, btc_price, "UP", 10)
            txt = bot.generate_content(client, 'REGULAR_TEXT', state, market_data=market_data)
            if txt:
                txt += f"\nSpot BTC: ${btc_price:,.0f}"
            else:
                txt = f"Bitcoin is ripping at ${btc_price:,.0f}!"
            bot.post_tweet(txt, image_path=chart_path)
        else:
            print("❌ Chart generation returned None.")
    else:
        print("❌ Error: function 'generate_chart' not found in bot.py")

if __name__ == "__main__":
    run_visual_simulation()
