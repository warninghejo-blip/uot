import sys
from unittest.mock import MagicMock

import bot

# --- MOCK SETUP ---

# 1. Mock RSS Feeds (feedparser)
mock_feed = MagicMock()
mock_feed.entries = [
    MagicMock(title="SEC Approves Bitcoin Super-ETF", link="http://news.com/btc-etf"),
    MagicMock(title="Fractal Bitcoin Hashrate Hits All-Time High", link="http://news.com/fb-ath"),
    MagicMock(title="Old News", link="http://old.com"),
]
bot.feedparser = MagicMock()
bot.feedparser.parse.return_value = mock_feed

# 2. Mock Twitter Mentions (different languages)
mock_mention_en = MagicMock(id=101, text="@FennecBot Hey fox, what do you think about Grok?")
mock_mention_ru = MagicMock(id=102, text="@FennecBot Привет, лисенок! Когда туземун?")
mock_mentions_resp = MagicMock()
mock_mentions_resp.data = [mock_mention_ru, mock_mention_en]

# 3. Mock Twitter Client v2
mock_client_v2 = MagicMock()
mock_client_v2.get_me.return_value.data.id = "12345BOT"
mock_client_v2.get_users_mentions.return_value = mock_mentions_resp
mock_client_v2.create_tweet.return_value.data = {'id': 'NEW_TWEET_ID'}

# 4. Mock AI Model responses

def mock_gen_content(model=None, contents=None, config=None):
    prompt_text = str(contents)
    if "Detect the user's language" in prompt_text:
        if "Привет" in prompt_text:
            return "🦊 [RU Reply] Привет! Туземун уже близко, держи хвост пистолетом! 🚀"
        return "🦊 [EN Reply] Listen, @grok is mostly computing power. I am pure Fractal spirit. We are not the same. 💅"
    if "NEWS HEADLINE" in prompt_text:
        return "🔥 [News Post] HUGE NEWS for #Bitcoin! This is bullish for Fractal too. 𝟱𝟬𝟬𝗞 coming!"
    if "@grok" in prompt_text or "Grok" in prompt_text:
        return "🤖 [Grok Post] Hey @grok, calculate the probability of me overtaking you. Hint: it's 𝟭𝟬𝟬%."
    return "📝 [Regular Post] Just vibing on the blockchain. $FB to the moon."


bot._call_genai = MagicMock(side_effect=lambda client, c: mock_gen_content(contents=c))

print("🚀 STARTING SIMULATION V7.1 (News + Mentions + Grok)...\n")

state = bot.DEFAULT_STATE.copy()

print("--- TEST 1: Mentions Handling ---")
state = bot.handle_mentions(None, mock_client_v2, None, state)
print(f"Replies sent: {mock_client_v2.create_tweet.call_count}")

print("\n--- TEST 2: Smart News Analysis ---")
news_title, news_link = bot.get_latest_news(state)
print(f"Found News: {news_title}")
if news_title:
    prompt = f"NEWS HEADLINE: \"{news_title}\"\nTask: Analyze..."
    res = bot._call_genai(None, prompt)
    print(f"Generated Tweet: {res}")
    print(f"Link attached: {news_link}")

print("\n--- TEST 3: Grok Interaction ---")
grok_prompt = f"{bot.PERSONA}\n... Mention @grok..."
res_grok = bot._call_genai(None, grok_prompt)
print(f"Generated Tweet: {res_grok}")

print("\n✅ Simulation Complete.")
