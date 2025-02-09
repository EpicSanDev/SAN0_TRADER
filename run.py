import threading
from web_interface import app, socketio
from trading_bot import GPTTradingBot
import logging
from datetime import datetime

def setup_logging():
    """Configure le système de logging"""
    try:
        log_filename = f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Erreur lors de la configuration du logging: {e}")

def run_trading_bot(bot):
    """Démarre le bot de trading"""
    try:
        bot.start()
    except Exception as e:
        logging.error(f"Erreur lors du démarrage du bot: {e}")

def run_web_interface(bot):
    """Démarre l'interface web"""
    try:
        # Initialize the trading bot instance in web interface
        from web_interface import init_trading_bot
        init_trading_bot(bot)
        socketio.run(app, debug=False, port=5000)
    except Exception as e:
        logging.error(f"Erreur lors du démarrage de l'interface web: {e}")

if __name__ == "__main__":
    # Configuration du logging
    setup_logging()
    logging.info("Démarrage de l'application...")

    try:
        # Create a single bot instance to be shared
        bot = GPTTradingBot()
        logging.info("Bot de trading initialisé")

        # Create threads with shared bot instance
        bot_thread = threading.Thread(target=run_trading_bot, args=(bot,))
        web_thread = threading.Thread(target=run_web_interface, args=(bot,))

        # Start threads
        logging.info("Démarrage du bot de trading...")
        bot_thread.start()
        logging.info("Démarrage de l'interface web...")
        web_thread.start()

        # Wait for threads to finish
        bot_thread.join()
        web_thread.join()

    except Exception as e:
        logging.error(f"Erreur lors du démarrage de l'application: {e}")
