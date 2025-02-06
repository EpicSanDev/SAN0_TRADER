# MetaTrader5 Configuration
MT5_LOGIN = "YOUR_MT5_LOGIN"  # Your MT5 account number
MT5_PASSWORD = "YOUR_MT5_PASSWORD"  # Your MT5 password
MT5_SERVER = "YOUR_MT5_SERVER"  # Your MT5 server name
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Your OpenAI API key

# Trading Parameters
SLIPPAGE = 10  # Maximum price deviation in points
SYMBOL = "EURUSD"  # Default trading symbol
TIMEFRAME = "M30"  # Default timeframe
LOT_SIZE = 0.01  # Default lot size
STOP_LOSS_PIPS = 50  # Default stop loss in pips
TAKE_PROFIT_PIPS = 100  # Default take profit in pips

# Trading Symbols
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "AUDUSD"]  # Add your trading pairs

# Risk Management
MAX_RISK_PERCENT = 1  # Maximum risk per trade (%)
MAX_DAILY_LOSS = 3  # Maximum daily loss (%)
MAX_TRADES_PER_DAY = 3  # Maximum trades per day

# Advanced Risk Management
RISK_MANAGEMENT = {
    "MAX_DRAWDOWN_PERCENT": 5,
    "DAILY_TARGET_PERCENT": 2,
    "WEEKLY_TARGET_PERCENT": 7,
    "POSITION_SCALING": True,
    "SCALE_OUT_LEVELS": [20, 40, 60],
    "MARTINGALE_FACTOR": 0.3,
    "CONSECUTIVE_LOSSES_LIMIT": 3,
    "PROFIT_LOCK": True,
    "PROFIT_LOCK_THRESHOLD": 1.5,
    "DYNAMIC_POSITION_SIZING": True
}

# Telegram Configuration (Optional)
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"  # Your Telegram bot token
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"  # Your Telegram chat ID

# Technical Parameters
TIMEFRAMES = ["M15", "H1", "H4"]
INDICATORS = ["RSI", "MACD", "MA"]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_PERIOD = 200

# Scoring Parameters
MIN_SCORE_TO_TRADE = 6
VOLATILITY_THRESHOLD = 0.3
TREND_STRENGTH_MIN = 35

# Market Filters
MARKET_HOURS = {
    "open": {"hour": 8, "minute": 0},
    "close": {"hour": 22, "minute": 0}
}
NEWS_IMPACT_THRESHOLD = "HIGH"  # LOW, MEDIUM, HIGH
MIN_DAILY_VOLUME = 1000

# Market Filters Configuration
MARKET_FILTERS = {
    "PRE_MARKET_HOURS": {"start": "07:30", "end": "08:00"},
    "MARKET_HOURS": {"start": "08:00", "end": "21:30"},
    "POST_MARKET_HOURS": {"start": "21:30", "end": "22:00"},
    "VOLATILITY_FILTER": True,
    "NEWS_FILTER": True,
    "SPREAD_FILTER": {
        "max_spread_pips": 3000,
        "spread_check_interval": 30
    },
    "VOLUME_FILTER": {
        "min_volume_threshold": 1500,
        "volume_ma_period": 20
    },
    "TREND_FILTER": {
        "min_trend_strength": 35,
        "trend_ma_periods": [20, 50, 200]
    }
}

# NewsAPI Configuration
NEWS_API_KEY = "YOUR_NEWS_API_KEY"  # Your NewsAPI key
NEWS_LOOKBACK_DAYS = 3
NEWS_LANGUAGES = ['en', 'fr']
NEWS_SENTIMENT_THRESHOLD = 0.3

# Add your symbol-specific news keywords
SYMBOL_NEWS_KEYWORDS = {
    "EURUSD": ["EUR/USD", "euro dollar", "ECB", "Federal Reserve"],
    "GBPUSD": ["GBP/USD", "pound sterling", "Bank of England", "UK economy"],
    # Add more symbols as needed
}

# Money Management
RISK_PER_TRADE_PERCENT = 0.3
MAX_LEVERAGE = 100
POSITION_SIZING_METHOD = "RISK"
MAX_POSITION_SIZE_PERCENT = 1.0
MIN_POSITION_SIZE = 0.01
COMPOUND_PROFITS = True
MARGIN_SAFETY_PERCENT = 70
RISK_REWARD_RATIO = 2.5
DYNAMIC_RISK_ADJUSTMENT = True
RISK_REDUCTION_AFTER_LOSS = 0.5
KELLY_FRACTION = 0.5
