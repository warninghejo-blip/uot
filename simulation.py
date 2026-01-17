import copy
import sys
import atexit
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock
import bot
import PIL.Image


LOG_PATH = Path(__file__).with_name("simulation_log.txt")
LOG_FILE = LOG_PATH.open('w', encoding='utf-8')


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


sys.stdout = _TeeStream(sys.__stdout__, LOG_FILE)
atexit.register(LOG_FILE.close)


def log_preview(scenario_name, label, content):
    timestamp = datetime.now().isoformat(timespec='seconds')
    safe_content = content if content else "[No content generated]"
    LOG_FILE.write(f"\n[{timestamp}] Scenario: {scenario_name} | {label}\n{safe_content}\n")
    LOG_FILE.flush()

# stats_text, fb_price, fennec_price, btc_price, trend_summary, fen_change
scenarios = [
    {
        "name": "🚀 MOON PUMP",
        "fb": 12.50,
        "fennec": 0.05,
        "btc": 105000,
        "trend_summary": "BTC: +5% | FB: +150% | FENNEC: +200%",
        "fen_change": 200.0
    },
    {
        "name": "🩸 CRASH",
        "fb": 2.10,
        "fennec": 0.0001,
        "btc": 85000,
        "trend_summary": "BTC: -10% | FB: -60% | FENNEC: -95%",
        "fen_change": -95.0
    },
    {
        "name": "😐 SIDEWAYS",
        "fb": 5.20,
        "fennec": 0.0040,
        "btc": 95000,
        "trend_summary": "BTC: +0.5% | FB: +1.2% | FENNEC: -0.5%",
        "fen_change": -0.5
    }
]

print("--- STARTING FENNEC ULTIMATE v6.0 SIMULATION ---")

# Mock setup_api to avoid real Twitter auth
bot.setup_api = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
model, *_ = bot.setup_api()

# Mock model generation
def mock_gen(prompt):
    # Detect if multimodal (list) or text (string)
    if isinstance(prompt, list):
        prompt_text = prompt[0]
    else:
        prompt_text = prompt
    return type("Resp", (), {"text": f"[MOCK AI OUTPUT for prompt type: {prompt_text[:150]}...]"})()

model.generate_content = MagicMock(side_effect=mock_gen)

# Mock PIL Image.open to avoid actual file operations
PIL.Image.open = MagicMock(return_value=MagicMock())

state_template = bot.load_state()

for sc in scenarios:
    print(f"\n\n>>> TESTING SCENARIO: {sc['name']}")
    
    market_data = (
        f"Simulated Stats: FB {sc['fb']}, FENNEC {sc['fennec']}", 
        sc['fb'], 
        sc['fennec'], 
        sc['btc'], 
        sc['trend_summary'], 
        sc['fen_change']
    )

    # 1. GM Post
    print("\n[GM Tweet Preview]:")
    gm_text = bot.generate_content(model, 'GM', state_template, market_data=market_data)
    print(gm_text)
    log_preview(sc['name'], 'GM Tweet Preview', gm_text)

    # 2. Burn Post (Vision Mock)
    print("\n[Burn (Vision) Preview]:")
    burn_text = bot.generate_content(model, 'BURN', state_template, context_data={'day': 205}, image_path="mock.png", market_data=market_data)
    print(burn_text)
    log_preview(sc['name'], 'Burn Preview', burn_text)

    # 3. Prophecy
    print("\n[Prophecy Preview]:")
    prophecy_text = bot.generate_content(model, 'PROPHECY', state_template, context_data={'candles': 'Close: 95000\nClose: 96000'}, market_data=market_data)
    print(prophecy_text)
    log_preview(sc['name'], 'Prophecy Preview', prophecy_text)

    # 4. Regular Post
    print("\n[Regular Post Preview]:")
    regular_text = bot.generate_content(model, 'REGULAR_TEXT', state_template, market_data=market_data)
    print(regular_text)
    log_preview(sc['name'], 'Regular Post Preview', regular_text)

completion_msg = "\n✅ Simulation complete. Results saved to simulation_log.txt"
print(completion_msg)
log_preview("GLOBAL", "Completion", completion_msg)
