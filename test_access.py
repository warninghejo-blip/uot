import os
import tweepy
from dotenv import load_dotenv

# Загружаем ключи из .env
load_dotenv()

def test_read_access():
    print("🕵️ ПРОВЕРКА ДОСТУПА НА ЧТЕНИЕ (X API v2)...")
    print("-" * 40)
    
    # Инициализация клиента
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_SECRET')
        )

        # 1. Проверка получения пользователя (Чтение)
        print("1. Запрашиваю данные @fractal_bitcoin...")
        user = client.get_user(username="fractal_bitcoin")
        
        if user.data:
            print(f"✅ УСПЕХ! ID пользователя: {user.data.id}")
            
            # 2. Проверка получения твитов (Чтение контента)
            print("2. Пробую прочитать последние твиты...")
            tweets = client.get_users_tweets(id=user.data.id, max_results=5)
            
            if tweets.data:
                print(f"✅ УСПЕХ! Найдено твитов: {len(tweets.data)}")
                print(f"📝 Последний твит: {tweets.data[0].text[:70]}...")
                return True
            else:
                print("⚠️ Доступ есть, но список твитов пуст.")
                return True
        else:
            print("❌ Ошибка: Пользователь не найден.")
            return False

    except tweepy.errors.Forbidden as e:
        print(f"\n❌ ОШИБКА 403 (FORBIDDEN): {e}")
        print("Вердикт: У вас 'Write-Only' доступ. Читать чужие твиты нельзя.")
        return False
    except Exception as e:
        print(f"\n❌ ПРОЧАЯ ОШИБКА: {e}")
        return False

if __name__ == "__main__":
    result = test_read_access()
    
    print("-" * 40)
    if result:
        print("🎉 ПОЗДРАВЛЯЮ! У ВАС ЕСТЬ ДОСТУП НА ЧТЕНИЕ!")
        print("Мы можем внедрять 'Умного Снайпера'.")
    else:
        print("🔒 Доступ ограничен только записью (Free Tier).")
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Ожидание ввода перед закрытием
    print("\n" + "="*40)
    input("Нажми ENTER, чтобы закрыть это окно...")