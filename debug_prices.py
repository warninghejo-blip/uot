import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = "https://fennec-api.warninghejo.workers.dev"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,fractal-bitcoin&vs_currencies=usd"
UNISAT_HOST = "https://open-api-fractal.unisat.io"

TICKER_FENNEC = "FENNEC"
TICKER_FB_POOL = "sFB___000"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def test_coingecko():
    print("\n🔎 ТЕСТ 1: CoinGecko (Прямой запрос)...")
    try:
        resp = requests.get(COINGECKO_URL, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Успех! Данные: {data}")
            fb = data.get('fractal-bitcoin', {}).get('usd', 0)
            print(f"💰 Цена FB: ${fb}")
        else:
            print(f"❌ Ошибка {resp.status_code}: {resp.text}")
    except Exception as exc:
        print(f"❌ Сбой запроса: {exc}")


def test_backend_pool():
    print(f"\n🔎 ТЕСТ 2: Backend quote ({TICKER_FENNEC}/{TICKER_FB_POOL})...")
    url = f"{BACKEND_URL}?action=quote&tick0={TICKER_FENNEC}&tick1={TICKER_FB_POOL}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        data = resp.json()
        print(f"RAW: {data}")

        pool_data = data.get('data', data)
        existed = pool_data.get('existed', False) if isinstance(pool_data, dict) else False

        if existed:
            amt0 = float(pool_data.get('amount0', 0))
            amt1 = float(pool_data.get('amount1', 0))
            print(f"✅ Пул найден. FENNEC: {amt0:,.2f}, FB: {amt1:,.2f}")
            if amt0 > 0:
                price_fb = amt1 / amt0
                print(f"🧮 1 FENNEC = {price_fb:.6f} FB")
        else:
            print("❌ Пул не найден (existed=false)")
            print_json(data)
    except Exception as exc:
        print(f"❌ Ошибка: {exc}")


def test_unisat_direct():
    print("\n🔎 ТЕСТ 3: UniSat API...")
    token = os.getenv('UNISAT_API_KEY')
    if not token:
        print("⚠️ Пропуск: нет UNISAT_API_KEY в .env")
        return

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    url = f"{UNISAT_HOST}/v3/market/swap/pool/list"
    try:
        resp = requests.get(url, headers=headers, params={'start': 0, 'limit': 200}, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Ошибка UniSat API {resp.status_code}: {resp.text}")
            return

        data = resp.json()
        pools = data.get('data', {}).get('list', [])

        found = False
        for pool in pools:
            ticks = {pool.get('tick0', '').upper(), pool.get('tick1', '').upper()}
            if 'FENNEC' in ticks:
                found = True
                print("✅ Пул найден в UniSat:")
                print_json(pool)
                break

        if not found:
            print("⚠️ Пул FENNEC не найден среди первых 200 результатов")
    except Exception as exc:
        print(f"❌ Ошибка UniSat: {exc}")


if __name__ == '__main__':
    test_coingecko()
    test_backend_pool()
    test_unisat_direct()
    input("\nНажми Enter, чтобы закрыть окно...")
