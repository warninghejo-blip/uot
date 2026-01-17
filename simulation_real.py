import os
import time
import logging
from unittest.mock import MagicMock
from dotenv import load_dotenv
from google import genai
import bot

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def run_real_simulation():
    print("\n🦊 INITIALIZING FENNEC REAL AI SIMULATION (v7.2)...")

    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env")
        return

    print("🧠 Connecting to Google Gemini API...", end=" ")
    try:
        real_model = genai.Client(api_key=api_key)
        print("SUCCESS ✅")
    except Exception as e:
        print(f"FAILED ❌\n{e}")
        return

    mock_api_v1 = MagicMock()
    mock_client_v2 = MagicMock()

    def print_tweet_instead_of_posting(api_v1, client_v2, text, **kwargs):
        print(f"\n📢 [WOULD POST TO TWITTER]:\n{'-' * 40}\n{text}\n{'-' * 40}")
        return "MOCK_TWEET_ID_123"

    bot.post_tweet = print_tweet_instead_of_posting

    state = bot.DEFAULT_STATE.copy()

    print("\n\n🧪 TEST 1: REAL NEWS ANALYSIS")
    print("(Giving Gemini a fake headline about Bitcoin ETF options)")

    fake_news_title = "SEC Officially Approves Options Trading for BlackRock's Bitcoin ETF"
    fake_news_link = "https://coindesk.com/fake-news"

    print(f"📥 Input News: {fake_news_title}")

    prompt_news = (
        f"{bot.PERSONA}\n\n"
        f"📰 NEWS HEADLINE: \"{fake_news_title}\"\n"
        "TASK: Analyze this news specifically for the Fractal Bitcoin ecosystem.\n"
        "STEPS:\n"
        "1. Identify the core event (Institutional liquidity).\n"
        "2. Explain why this matters for Fractal (L2).\n"
        "3. Write a high-IQ, punchy tweet.\n"
        "STYLE: Use 1-2 bullet points. Add a sarcastic or confident remark.\n"
        "OUTPUT: Only the tweet text."
    )

    start_t = time.time()
    response = bot._call_genai(real_model, prompt_news)
    print(f"⏱️ Generation time: {time.time() - start_t:.2f}s")

    bot.post_tweet(None, None, f"{response}\n{fake_news_link}")

    print("\n\n🧪 TEST 2: RUSSIAN USER INTERACTION")
    print("(User asks: 'Should I sell Fennec?')")

    mock_mention_ru = MagicMock()
    mock_mention_ru.id = 101
    mock_mention_ru.text = "@FennecBot Брат, цена падает! Сливать FENNEC или держать? Мне страшно."

    mock_client_v2.get_users_mentions.return_value.data = [mock_mention_ru]
    mock_client_v2.get_me.return_value.data.id = "BOT_ID"

    bot.handle_mentions(mock_api_v1, mock_client_v2, real_model, state)

    print("\n\n🧪 TEST 3: ENGLISH GROK INTERACTION")
    print("(User says: 'Grok is smarter than you')")

    mock_mention_en = MagicMock()
    mock_mention_en.id = 102
    mock_mention_en.text = "@FennecBot I think @grok is way smarter than a JPEG fox. You are outdated."

    mock_client_v2.get_users_mentions.return_value.data = [mock_mention_en]

    bot.handle_mentions(mock_api_v1, mock_client_v2, real_model, state)

    print("\n\n🧪 TEST 4: REGULAR SHITPOST (Random Topic)")

    market_data = ("BTC: $98,000", 5.50, 0.006, 98000.0, "Trend: UP", 5.0)

    content = bot.generate_content(real_model, 'REGULAR_TEXT', state, market_data=market_data)
    bot.post_tweet(None, None, content)


if __name__ == "__main__":
    run_real_simulation()
