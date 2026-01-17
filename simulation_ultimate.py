import os
import time
import logging
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from google import genai

import bot


logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')


def mock_post_tweet(api_v1, client_v2, text, image_path=None, **kwargs):
    print(f"\n🐦 [POST] {'(with image)' if image_path else ''}\n{text}\n")
    if image_path:
        print(f"📎 Image: {image_path}")
    return f"SIM_{int(time.time())}"


@dataclass
class FakeMention:
    id: int
    text: str


class FakeMentionsResponse:
    def __init__(self, mentions):
        self.data = [FakeMention(i + 1, txt) for i, txt in enumerate(mentions)]


class FakeUser:
    def __init__(self, user_id):
        self.data = type('obj', (), {'id': user_id})


class FakeClientV2:
    def __init__(self, mentions):
        self._mentions = mentions

    def get_me(self):
        return FakeUser(42)

    def get_users_mentions(self, *args, **kwargs):
        return FakeMentionsResponse(self._mentions)


def setup_state():
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('Missing GEMINI_API_KEY')
    client = genai.Client(api_key=api_key)
    state = bot.DEFAULT_STATE.copy()
    return client, state


def simulate_mentions(state, model_client):
    print('\n=== STEP 1: Mentions ===')
    mentions = [
        "@FennecBot Привет, как дела?",
        "@FennecBot Hey, @grok thinks he can out-trade you."
    ]
    fake_client = FakeClientV2(mentions)
    state['bot_user_id'] = 42
    bot.post_tweet = mock_post_tweet
    updated_state = bot.handle_mentions(None, fake_client, model_client, state)
    print('✅ Mentions processed')
    return updated_state


def simulate_news(state, model_client):
    print('\n=== STEP 2: News Analysis ===')
    news_title = "SpaceX adds Fractal Bitcoin to payment options"
    news_link = "https://newsroom.spacex.com/fractal"
    prompt = (
        f"{bot.PERSONA}\n\n"
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
    content = bot._call_genai(model_client, prompt)
    news_tweet = f"{content}\n{news_link}" if content else 'News generation failed.'
    print(news_tweet)
    return news_tweet


def fetch_real_history(coin_id):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    prices = data.get('prices', [])
    if not prices:
        raise RuntimeError(f"No data returned for {coin_id}")
    return [{'timestamp': ts / 1000, 'price': price} for ts, price in prices]


def simulate_chart(state, model_client, coin_name, coin_symbol, coin_id, column_key):
    print(f"\n=== REAL CHART TEST: {coin_name.upper()} ===")
    raw_history = fetch_real_history(coin_id)
    history = []
    for entry in raw_history:
        point = {'timestamp': entry['timestamp'], column_key: entry['price']}
        history.append(point)
    state['price_history'] = history

    chart_path = bot.generate_chart(state, coin=column_key)
    if not chart_path:
        raise RuntimeError(f"Chart generation failed for {coin_name}")

    summary, changes = bot.calculate_trend(state)
    change_pct = changes.get(column_key, 0.0)
    last_price = raw_history[-1]['price']

    stats_text = f"{coin_name} spot: ${last_price:,.2f}"
    fb_price = last_price if column_key == 'fb' else 0.0
    fennec_price = last_price if column_key == 'fennec' else 0.0
    btc_price = last_price if column_key == 'btc' else 0.0
    market_data = (stats_text, fb_price, fennec_price, btc_price, summary, changes)

    caption = bot.generate_content(
        model_client,
        'CHART_CAPTION',
        state,
        context_data={'coin': coin_symbol, 'change': change_pct, 'price': last_price},
        market_data=market_data
    )

    print(f"Generated {coin_symbol} caption: {caption}")
    bot.post_tweet(None, None, caption or f'{coin_symbol} move!', image_path=chart_path)
    state['last_chart_post_time'] = time.time()
    state['last_any_post_time'] = state['last_chart_post_time']
    print(f"Chart saved at {chart_path}, 24h change={change_pct:+.2f}%")
    return caption


def simulate_rate_limit(state):
    print('\n=== STEP 4: Global Rate Limit Test ===')
    delta = time.time() - state.get('last_any_post_time', 0)
    if delta < bot.GLOBAL_POST_INTERVAL:
        wait = bot.GLOBAL_POST_INTERVAL - delta
        print(f"⏳ Regular post blocked. Need to wait {wait/60:.1f} more minutes.")
    else:
        print('Rate limiter expired — post would proceed (not expected in this demo).')


def main():
    client, state = setup_state()
    simulate_mentions(state, client)
    simulate_news(state, client)
    simulate_chart(state, client, 'Bitcoin', 'BTC', 'bitcoin', 'btc')
    state['last_any_post_time'] = 0  # reset to show second chart demo
    simulate_chart(state, client, 'Fractal Bitcoin', 'FB', 'fractal-bitcoin', 'fb')
    print('\nSkipped FENNEC chart in demo to avoid synthetic pricing.')
    simulate_rate_limit(state)


if __name__ == '__main__':
    main()
