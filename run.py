import threading
from web_interface import app, socketio
from trading_bot import GPTTradingBot
import logging
from datetime import datetime
import time

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

def run_telegram_bot(bot):
    """Gère les updates Telegram en continu"""
    try:
        last_update_id = 0
        while True:
            try:
                # Récupérer les updates avec un long polling
                updates = bot.telegram_bot.get_updates(
                    offset=last_update_id + 1,
                    timeout=30,  # Long polling de 30 secondes
                    allowed_updates=['message']
                )
                
                for update in updates:
                    if update.message and update.message.text:
                        if update.message.text.startswith('/'):
                            command_parts = update.message.text[1:].split()
                            command = command_parts[0]
                            args = command_parts[1:] if len(command_parts) > 1 else []
                            bot.process_telegram_command(command, *args)
                    last_update_id = update.update_id
                    
            except Exception as e:
                logging.error(f"Erreur lors de la gestion des updates Telegram: {e}")
                time.sleep(5)  # Attendre un peu avant de réessayer
                
    except Exception as e:
        logging.error(f"Erreur fatale dans le thread Telegram: {e}")

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
        telegram_thread = threading.Thread(target=run_telegram_bot, args=(bot,))

        # Start threads
        logging.info("Démarrage du bot de trading...")
        bot_thread.start()
        logging.info("Démarrage de l'interface web...")
        web_thread.start()
        logging.info("Démarrage du bot Telegram...")
        telegram_thread.start()

        # Wait for threads to finish
        bot_thread.join()
        web_thread.join()
        telegram_thread.join()

    except Exception as e:
        logging.error(f"Erreur lors du démarrage de l'application: {e}")
