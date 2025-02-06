# Assurez-vous que le login est un nombre valide
MT5_LOGIN = ""  # Remplacez par votre numéro de compte réel
MT5_PASSWORD = ""  # Remplacez par votre mot de passe réel
MT5_SERVER = ""  # Vérifiez le nom exact du serveur
OPENROUTER_API_KEY = ""  # Remplacez par votre clé API OpenRouter

# Paramètres de trading
SLIPPAGE = 10  # Déviation maximale du prix en points
SYMBOL = "EURUSD"
TIMEFRAME = "M30"
LOT_SIZE = 0.01
STOP_LOSS_PIPS = 50
TAKE_PROFIT_PIPS = 100

# Liste des symboles à trader
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD", "AUDUSD","NZDUSD","USDCAD","USDCHF","EURGBP","EURJPY","EURCHF","GBPJPY","GBPCHF","AUDJPY","AUDNZD","AUDCAD","AUDCHF","NZDJPY","NZDCHF","CADJPY","CADCHF","CHFJPY","EURAUD","GBPAUD"]

# Paramètres de gestion des risques
MAX_RISK_PERCENT = 1  # Réduction du risque par trade
MAX_DAILY_LOSS = 3    # Réduction de la perte maximale journalière
MAX_TRADES_PER_DAY = 3 # Réduction du nombre de trades quotidiens

# Nouveaux paramètres de gestion des risques avancée
RISK_MANAGEMENT = {
    "MAX_DRAWDOWN_PERCENT": 5,     # Réduction du drawdown maximum
    "DAILY_TARGET_PERCENT": 2,     # Objectif journalier plus conservateur
    "WEEKLY_TARGET_PERCENT": 7,    # Objectif hebdomadaire plus conservateur
    "POSITION_SCALING": True,      
    "SCALE_OUT_LEVELS": [20, 40, 60], # Prises de profits plus rapides
    "MARTINGALE_FACTOR": 0.3,     # Facteur plus conservateur
    "CONSECUTIVE_LOSSES_LIMIT": 3, # Arrêt après 3 pertes consécutives
    "PROFIT_LOCK": True,          # Verrouillage des profits
    "PROFIT_LOCK_THRESHOLD": 1.5,  # Seuil de verrouillage en %
    "DYNAMIC_POSITION_SIZING": True # Ajustement dynamique des positions
}

# Paramètres Telegram
TELEGRAM_TOKEN = "8130439939:AAG0asyQEXvytzAB3QZkHrbkkkCv7Q0hXeM"
TELEGRAM_CHAT_ID = "-4546591195"

# Paramètres techniques supplémentaires
TIMEFRAMES = ["M15", "H1", "H4"]  # Multiple timeframes
INDICATORS = ["RSI", "MACD", "MA"]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MA_PERIOD = 200

# Paramètres de scoring
MIN_SCORE_TO_TRADE = 6  # Score minimum plus élevé
VOLATILITY_THRESHOLD = 0.3  # Seuil de volatilité plus strict
TREND_STRENGTH_MIN = 35  # Force minimale de tendance plus élevée

# Paramètres de filtres de marché
MARKET_HOURS = {
    "open": {"hour": 8, "minute": 0},
    "close": {"hour": 22, "minute": 0}
}
NEWS_IMPACT_THRESHOLD = "HIGH"  # LOW, MEDIUM, HIGH
MIN_DAILY_VOLUME = 1000  # Changer de 100000 à 1000 pour être moins restrictif

# Filtres de marché améliorés
MARKET_FILTERS = {
    "PRE_MARKET_HOURS": {"start": "07:30", "end": "08:00"},
    "MARKET_HOURS": {"start": "08:00", "end": "21:30"},
    "POST_MARKET_HOURS": {"start": "21:30", "end": "22:00"},
    "VOLATILITY_FILTER": True,
    "NEWS_FILTER": True,
    "SPREAD_FILTER": {
        "max_spread_pips": 3000,  # Spread maximum plus strict
        "spread_check_interval": 30  # Vérification plus fréquente
    },
    "VOLUME_FILTER": {
        "min_volume_threshold": 1500,
        "volume_ma_period": 20
    },
    "TREND_FILTER": {
        "min_trend_strength": 35,
        "trend_ma_periods": [20, 50, 200]
    },
    "CORRELATION_FILTER": {
        "min_correlation": 0.8,
        "lookback_period": 50
    }
}

# Paramètres de backtesting
BACKTEST_PERIOD_DAYS = 30
OPTIMIZATION_ITERATIONS = 1000

# Paramètres de performance
CACHE_DURATION = 300  # Durée du cache en secondes
MAX_API_RETRIES = 3
EXECUTION_TIMEOUT = 10

# Paramètres de TP/SL automatique
AUTO_SL_ATR_MULTIPLIER = 1.2    # Stop Loss plus serré
AUTO_TP_ATR_MULTIPLIER = 2.0    # Take Profit plus conservateur
MIN_SL_PIPS = 15               # Stop Loss minimum réduit
MAX_SL_PIPS = 50              # Stop Loss maximum réduit
MIN_TP_PIPS = 25              # Take Profit minimum ajusté
MAX_TP_PIPS = 100             # Take Profit maximum réduit
USE_AUTO_TPSL = True
USE_TRAILING_STOP = True       # Activation du trailing stop
TRAILING_STOP_START = 0.5      # Début du trailing à 0.5% de profit
TRAILING_STOP_STEP = 0.2       # Pas du trailing stop en %

# Paramètres d'analyse multiple
CORRELATION_THRESHOLD = 0.7  # Seuil de corrélation entre actifs
MARKET_SENTIMENT_WEIGHT = 0.3  # Poids du sentiment de marché
INTERMARKET_ANALYSIS = {
    "EURUSD": ["GBPUSD", "USDJPY", "EURGBP", "EURJPY", "EURCHF", "EURAUD"],
    "GBPUSD": ["EURUSD", "USDJPY", "EURGBP", "GBPJPY", "GBPCHF", "GBPAUD"],
    "USDJPY": ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY"],
    "XAUUSD": ["USDJPY", "BTCUSD", "USDCHF"],
    "BTCUSD": ["XAUUSD"],
    "AUDUSD": ["NZDUSD", "EURAUD", "GBPAUD", "AUDNZD", "AUDCAD", "AUDJPY"],
    "NZDUSD": ["AUDUSD", "AUDNZD", "NZDJPY", "NZDCHF"],
    "USDCAD": ["AUDCAD", "CADJPY", "CADCHF"],
    "USDCHF": ["EURCHF", "GBPCHF", "AUDCHF", "CADCHF", "CHFJPY"],
    "EURGBP": ["EURUSD", "GBPUSD", "EURJPY", "GBPJPY"],
    "EURJPY": ["USDJPY", "EURGBP", "GBPJPY", "AUDJPY"],
    "EURCHF": ["USDCHF", "GBPCHF", "AUDCHF"],
    "GBPJPY": ["USDJPY", "EURJPY", "AUDJPY"],
    "GBPCHF": ["USDCHF", "EURCHF"],
    "AUDJPY": ["USDJPY", "EURJPY", "GBPJPY"],
    "AUDNZD": ["AUDUSD", "NZDUSD"],
    "AUDCAD": ["AUDUSD", "USDCAD"],
    "AUDCHF": ["AUDUSD", "USDCHF", "EURCHF"],
    "NZDJPY": ["USDJPY", "NZDUSD", "AUDJPY"],
    "NZDCHF": ["NZDUSD", "USDCHF"],
    "CADJPY": ["USDJPY", "USDCAD"],
    "CADCHF": ["USDCAD", "USDCHF"],
    "CHFJPY": ["USDJPY", "USDCHF"],
    "EURAUD": ["EURUSD", "AUDUSD"],
    "GBPAUD": ["GBPUSD", "AUDUSD"]
}

# Paramètres d'analyse technique avancés
TECHNICAL_INDICATORS = {
    "RSI": {"period": 14, "overbought": 70, "oversold": 30},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "MA": {"periods": [20, 50, 200]},
    "BB": {"period": 20, "std_dev": 2},
    "VWAP": {"period": "D"},
    "ATR": {"period": 14},
    "Momentum": {"period": 14},
    "OBV": {},
}

# Paramètres d'analyse avancés
ADVANCED_ANALYSIS = {
    "USE_MACHINE_LEARNING": True,
    "MARKET_REGIME_DETECTION": True,
    "SENTIMENT_ANALYSIS": True,
    "VOLATILITY_REGIMES": {
        "LOW": 10,
        "MEDIUM": 20,
        "HIGH": 30
    }
}

# Configuration des patterns de chandeliers japonais
CANDLESTICK_PATTERNS = {
    "RECOGNITION": {
        "DOJI_TOLERANCE": 0.1,  # Tolérance pour la taille du corps du Doji
        "HAMMER_BODY_RATIO": 0.3,  # Ratio corps/mèche pour les marteaux
        "ENGULFING_MIN_SIZE": 1.5,  # Taille minimum pour un pattern englobant
        "STAR_GAP_MINIMUM": 0.5,  # Écart minimum pour les étoiles
    },
    "VALIDATION": {
        "MIN_PATTERN_SCORE": 6,  # Score minimum pour considérer un pattern
        "CONFLUENCE_REQUIRED": 2,  # Nombre de confirmations requises
        "VOLUME_CONFIRMATION": True,  # Exiger une confirmation du volume
    },
    "WEIGHTS": {
        "DOJI": 0.6,
        "HAMMER": 0.7,
        "ENGULFING": 0.8,
        "MORNING_STAR": 0.9,
        "EVENING_STAR": 0.9,
        "THREE_WHITE_SOLDIERS": 1.0,
        "THREE_BLACK_CROWS": 1.0,
    }
}

# Critères de confirmation
MIN_CONFIRMATION_SIGNALS = 2  # Réduction du nombre de signaux requis
TREND_CONFIRMATION_PERIOD = 3  # Nombre de périodes pour confirmer tendance

# Stratégie de trading
TRADING_STRATEGY = {
    "ENTRY_RULES": {
        "TREND_FOLLOWING": {
            "description": "Stratégie de suivi de tendance",
            "conditions": {
                "trend_alignment": {
                    "timeframes": ["H4", "H1", "M15"],
                    "min_aligned": 2,
                    "weight": 0.3
                },
                "momentum": {
                    "rsi_conditions": {
                        "oversold_threshold": 30,
                        "overbought_threshold": 70,
                        "trend_confirmation": True
                    },
                    "macd_conditions": {
                        "histogram_direction": True,
                        "signal_cross": True
                    },
                    "weight": 0.2
                },
                "volume_confirmation": {
                    "min_volume_threshold": 1.2,
                    "volume_trend_aligned": True,
                    "weight": 0.15
                },
                "price_action": {
                    "candlestick_patterns": True,
                    "support_resistance": True,
                    "weight": 0.2
                },
                "market_context": {
                    "volatility_check": True,
                    "spread_check": True,
                    "news_impact": True,
                    "weight": 0.15
                }
            },
            "validation": {
                "min_conditions_met": 3,
                "min_total_score": 7.5,
                "confirmation_timeframe": "H1"
            }
        },
        "BREAKOUT": {
            "description": "Stratégie de breakout",
            "conditions": {
                "price_breakout": {
                    "key_level_types": ["support", "resistance", "psychological"],
                    "confirmation_candles": 2,
                    "weight": 0.35
                },
                "volume_surge": {
                    "min_volume_increase": 1.5,
                    "sustain_periods": 2,
                    "weight": 0.25
                },
                "volatility_expansion": {
                    "atr_multiplier": 1.2,
                    "bb_width_increase": 1.3,
                    "weight": 0.2
                },
                "momentum_confirmation": {
                    "rsi_direction": True,
                    "macd_alignment": True,
                    "weight": 0.2
                }
            },
            "validation": {
                "min_conditions_met": 3,
                "min_total_score": 8.0,
                "confirmation_timeframe": "M15"
            }
        }
    },
    "EXIT_RULES": {
        "TAKE_PROFIT": {
            "methods": {
                "fixed": {
                    "r_multiple": 2.0,
                    "enabled": True
                },
                "fibonacci": {
                    "levels": [1.618, 2.618, 4.236],
                    "enabled": True
                },
                "volatility_based": {
                    "atr_multiplier": 3.0,
                    "enabled": True
                }
            },
            "partial_exits": {
                "enabled": True,
                "levels": [
                    {"profit": 1.0, "size": 0.3},
                    {"profit": 1.5, "size": 0.3},
                    {"profit": 2.0, "size": 0.4}
                ]
            }
        },
        "STOP_LOSS": {
            "methods": {
                "fixed": {
                    "r_multiple": 1.0,
                    "enabled": True
                },
                "volatility_based": {
                    "atr_multiplier": 1.5,
                    "enabled": True
                },
                "swing_based": {
                    "lookback_periods": 5,
                    "enabled": True
                }
            },
            "trailing_stop": {
                "enabled": True,
                "activation_threshold": 1.0,
                "step_size": 0.5
            }
        },
        "POSITION_MANAGEMENT": {
            "max_holding_time": {
                "hours": 48,
                "enabled": True
            },
            "breakeven_stop": {
                "profit_threshold": 0.8,
                "enabled": True
            },
            "time_based_exit": {
                "end_of_day": True,
                "before_news": True,
                "weekend_close": True
            }
        }
    },
    "RISK_MANAGEMENT": {
        "position_sizing": {
            "method": "risk_based",
            "max_risk_per_trade": 0.01,
            "max_position_size": 0.02,
            "account_risk_factor": 0.8
        },
        "exposure_limits": {
            "max_daily_trades": 3,
            "max_concurrent_trades": 2,
            "max_correlation": 0.7,
            "max_sector_exposure": 0.2
        },
        "drawdown_control": {
            "max_daily_loss": 0.02,
            "max_weekly_loss": 0.05,
            "trailing_drawdown": 0.1
        }
    }
}

# Système de scoring avancé
SCORING_SYSTEM = {
    "TECHNICAL_WEIGHT": 0.45,    # Plus de poids sur l'analyse technique
    "FUNDAMENTAL_WEIGHT": 0.15,   # Moins de poids sur le fondamental
    "SENTIMENT_WEIGHT": 0.25,     # Plus de poids sur le sentiment
    "VOLATILITY_WEIGHT": 0.15,    # Moins de poids sur la volatilité
    "MIN_TOTAL_SCORE": 8.0,      # Score minimum plus élevé
    "CONFIRMATION_LEVELS": [
        {"score": 8.5, "risk_factor": 0.8},  # Plus conservateur
        {"score": 9.0, "risk_factor": 1.0},
        {"score": 9.5, "risk_factor": 1.2}
    ],
    "TREND_ALIGNMENT_BONUS": 0.2, # Bonus pour l'alignement des timeframes
    "VOLUME_CONFIRMATION_BONUS": 0.15, # Bonus pour la confirmation du volume
    "PATTERN_CONFIRMATION_BONUS": 0.1  # Bonus pour les patterns confirmés
}

# Paramètres de Money Management
RISK_PER_TRADE_PERCENT = 0.3     # Risque par trade très réduit
MAX_LEVERAGE = 100               # Levier maximum très réduit
POSITION_SIZING_METHOD = "RISK"  
MAX_POSITION_SIZE_PERCENT = 1.0  # Taille maximale très réduite
MIN_POSITION_SIZE = 0.01
STOP_LOSS_PIPS = 30             # Stop loss plus serré
COMPOUND_PROFITS = True
MARGIN_SAFETY_PERCENT = 70      # Marge de sécurité augmentée
RISK_REWARD_RATIO = 2.5         # Ratio risque/récompense augmenté
DYNAMIC_RISK_ADJUSTMENT = True  # Ajustement dynamique du risque
RISK_REDUCTION_AFTER_LOSS = 0.5 # Réduction du risque après une perte
KELLY_FRACTION = 0.5  # Fraction de Kelly pour le sizing (0.5 = demi-Kelly plus conservateur)

# Configuration NewsAPI
NEWS_API_KEY = ""
NEWS_LOOKBACK_DAYS = 3
NEWS_LANGUAGES = ['en', 'fr']
NEWS_SENTIMENT_THRESHOLD = 0.3

# Mapping des symboles vers les mots-clés de recherche
SYMBOL_NEWS_KEYWORDS = {
    "EURUSD": ["EUR/USD", "euro dollar", "ECB", "Federal Reserve", "European Union economy"],
    "GBPUSD": ["GBP/USD", "pound sterling", "Bank of England", "Brexit", "UK economy"],
    "USDJPY": ["USD/JPY", "yen dollar", "Bank of Japan", "Japanese economy"],
    "XAUUSD": ["gold price", "gold market", "precious metals", "gold trading"],
    "BTCUSD": ["bitcoin", "crypto", "cryptocurrency", "BTC"],
    "AUDUSD": ["AUD/USD", "Australian dollar", "RBA", "Australian economy"],
    "NZDUSD": ["NZD/USD", "New Zealand dollar", "RBNZ", "New Zealand economy"],
    "USDCAD": ["USD/CAD", "Canadian dollar", "Bank of Canada", "Canadian economy", "oil prices"],
    "USDCHF": ["USD/CHF", "Swiss franc", "SNB", "Swiss economy"],
    "EURGBP": ["EUR/GBP", "euro pound", "ECB", "Bank of England", "Brexit"],
    "EURJPY": ["EUR/JPY", "euro yen", "ECB", "Bank of Japan"],
    "EURCHF": ["EUR/CHF", "euro franc", "ECB", "SNB"],
    "GBPJPY": ["GBP/JPY", "pound yen", "Bank of England", "Bank of Japan"],
    "GBPCHF": ["GBP/CHF", "pound franc", "Bank of England", "SNB"],
    "AUDJPY": ["AUD/JPY", "aussie yen", "RBA", "Bank of Japan"],
    "AUDNZD": ["AUD/NZD", "aussie kiwi", "RBA", "RBNZ"],
    "AUDCAD": ["AUD/CAD", "aussie loonie", "RBA", "Bank of Canada"],
    "AUDCHF": ["AUD/CHF", "aussie franc", "RBA", "SNB"],
    "NZDJPY": ["NZD/JPY", "kiwi yen", "RBNZ", "Bank of Japan"],
    "NZDCHF": ["NZD/CHF", "kiwi franc", "RBNZ", "SNB"],
    "CADJPY": ["CAD/JPY", "loonie yen", "Bank of Canada", "Bank of Japan"],
    "CADCHF": ["CAD/CHF", "loonie franc", "Bank of Canada", "SNB"],  
    "CHFJPY": ["CHF/JPY", "franc yen", "SNB", "Bank of Japan"],
    "EURAUD": ["EUR/AUD", "euro aussie", "ECB", "RBA"],
    "GBPAUD": ["GBP/AUD", "pound aussie", "Bank of England", "RBA"]
}
