import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import time
import logging
import telegram
from datetime import datetime, timedelta
from config import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import json
import re
from news_analyzer import NewsAnalyzer
import joblib
from sklearn.ensemble import RandomForestClassifier
from economic_calendar import ForexFactoryCalendar
from requests import post
import os
class CandlePattern:
    def __init__(self, name, bullish, significance):
        self.name = name
        self.bullish = bullish  # True pour haussier, False pour baissier
        self.significance = significance  # 1-10

    def __str__(self):
        return f"{self.name} ({'Bullish' if self.bullish else 'Bearish'}) - Score: {self.significance}/10"

class PatternRecognizer:
    def __init__(self):
        self.patterns = {
            'doji': self.is_doji,
            'hammer': self.is_hammer,
            'engulfing': self.is_engulfing,
            'morning_star': self.is_morning_star,
            'evening_star': self.is_evening_star,
            'three_white_soldiers': self.is_three_white_soldiers,
            'three_black_crows': self.is_three_black_crows,
            'piercing_line': self.is_piercing_line,
            'dark_cloud_cover': self.is_dark_cloud_cover,
        }

    def analyze_candle_patterns(self, df, lookback=5):
        """Analyse les dernières bougies pour identifier les patterns"""
        patterns_found = []
        recent_data = df.tail(lookback)
        
        for pattern_name, pattern_func in self.patterns.items():
            if pattern_func(recent_data):
                pattern = self.get_pattern_details(pattern_name, recent_data)
                if pattern:
                    patterns_found.append(pattern)
        
        return patterns_found

    def is_doji(self, df):
        """Identifie un Doji"""
        latest = df.iloc[-1]
        body_size = abs(latest['open'] - latest['close'])
        wick_size = latest['high'] - latest['low']
        return body_size <= (wick_size * 0.1)

    def is_hammer(self, df):
        """Identifie un marteau ou un marteau inversé"""
        latest = df.iloc[-1]
        body_size = abs(latest['open'] - latest['close'])
        upper_wick = latest['high'] - max(latest['open'], latest['close'])
        lower_wick = min(latest['open'], latest['close']) - latest['low']
        
        # Pour un marteau classique
        if lower_wick > (body_size * 2) and upper_wick < body_size:
            return True
        # Pour un marteau inversé
        if upper_wick > (body_size * 2) and lower_wick < body_size:
            return True
        return False

    def is_engulfing(self, df):
        """Identifie un pattern englobant"""
        if len(df) < 2:
            return False
            
        prev = df.iloc[-2]
        curr = df.iloc[-1]
        
        prev_body_size = abs(prev['open'] - prev['close'])
        curr_body_size = abs(curr['open'] - curr['close'])
        
        # Bullish engulfing
        if (prev['close'] < prev['open'] and  # Previous bearish
            curr['close'] > curr['open'] and  # Current bullish
            curr['open'] < prev['close'] and 
            curr['close'] > prev['open']):
            return True
            
        # Bearish engulfing
        if (prev['close'] > prev['open'] and  # Previous bullish
            curr['close'] < curr['open'] and  # Current bearish
            curr['open'] > prev['close'] and 
            curr['close'] < prev['open']):
            return True
            
        return False

    def get_pattern_details(self, pattern_name, data):
        """Retourne les détails d'un pattern identifié"""
        pattern_details = {
            'doji': CandlePattern('Doji', True, 6),
            'hammer': CandlePattern('Hammer', True, 7),
            'engulfing': CandlePattern('Engulfing', data.iloc[-1]['close'] > data.iloc[-1]['open'], 8),
            'morning_star': CandlePattern('Morning Star', True, 9),
            'evening_star': CandlePattern('Evening Star', False, 9),
            'three_white_soldiers': CandlePattern('Three White Soldiers', True, 10),
            'three_black_crows': CandlePattern('Three Black Crows', False, 10),
            'piercing_line': CandlePattern('Piercing Line', True, 7),
            'dark_cloud_cover': CandlePattern('Dark Cloud Cover', False, 7),
        }
        return pattern_details.get(pattern_name)

    def is_morning_star(self, df):
        """Identifie un pattern Morning Star"""
        if len(df) < 3:
            return False
            
        candle1 = df.iloc[-3]  # Première bougie baissière
        candle2 = df.iloc[-2]  # Petite bougie (doji)
        candle3 = df.iloc[-1]  # Bougie haussière
        
        # Vérification première bougie baissière
        if not (candle1['close'] < candle1['open']):
            return False
            
        # Vérification du doji ou petite bougie
        body_size2 = abs(candle2['close'] - candle2['open'])
        if body_size2 > (candle1['high'] - candle1['low']) * 0.3:
            return False
            
        # Vérification troisième bougie haussière
        if not (candle3['close'] > candle3['open']):
            return False
            
        # Vérification des gaps
        if not (candle2['high'] < candle1['low'] and candle2['low'] < candle3['open']):
            return False
            
        return True

    def is_evening_star(self, df):
        """Identifie un pattern Evening Star"""
        if len(df) < 3:
            return False
            
        candle1 = df.iloc[-3]  # Première bougie haussière
        candle2 = df.iloc[-2]  # Petite bougie (doji)
        candle3 = df.iloc[-1]  # Bougie baissière
        
        # Vérification première bougie haussière
        if not (candle1['close'] > candle1['open']):
            return False
            
        # Vérification du doji ou petite bougie
        body_size2 = abs(candle2['close'] - candle2['open'])
        if body_size2 > (candle1['high'] - candle1['low']) * 0.3:
            return False
            
        # Vérification troisième bougie baissière
        if not (candle3['close'] < candle3['open']):
            return False
            
        # Vérification des gaps
        if not (candle2['low'] > candle1['high'] and candle2['high'] > candle3['open']):
            return False
            
        return True

    def is_three_white_soldiers(self, df):
        """Identifie un pattern Three White Soldiers"""
        if len(df) < 3:
            return False
            
        for i in range(-3, 0):
            candle = df.iloc[i]
            if not (candle['close'] > candle['open']):  # Bougie haussière
                return False
            
            # Vérification de la progression
            if i < -1:
                next_candle = df.iloc[i + 1]
                if not (candle['close'] < next_candle['close'] and 
                       candle['open'] < next_candle['open']):
                    return False
                    
            # Vérification des mèches
            body_size = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - candle['close']
            if upper_wick > body_size * 0.2:  # Mèche supérieure courte
                return False
                
        return True

    def is_three_black_crows(self, df):
        """Identifie un pattern Three Black Crows"""
        if len(df) < 3:
            return False
            
        for i in range(-3, 0):
            candle = df.iloc[i]
            if not (candle['close'] < candle['open']):  # Bougie baissière
                return False
            
            # Vérification de la progression
            if i < -1:
                next_candle = df.iloc[i + 1]
                if not (candle['close'] > next_candle['close'] and 
                       candle['open'] > next_candle['open']):
                    return False
                    
            # Vérification des mèches
            body_size = abs(candle['close'] - candle['open'])
            lower_wick = candle['close'] - candle['low']
            if lower_wick > body_size * 0.2:  # Mèche inférieure courte
                return False
                
        return True

    def is_piercing_line(self, df):
        """Identifie un pattern Piercing Line"""
        if len(df) < 2:
            return False
            
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]
        
        # Première bougie baissière
        if not (prev_candle['close'] < prev_candle['open']):
            return False
            
        # Deuxième bougie haussière
        if not (curr_candle['close'] > curr_candle['open']):
            return False
            
        # Ouverture en gap baissier
        if not (curr_candle['open'] < prev_candle['low']):
            return False
            
        # Clôture au-dessus du milieu de la première bougie
        prev_midpoint = (prev_candle['open'] + prev_candle['close']) / 2
        if not (curr_candle['close'] > prev_midpoint):
            return False
            
        return True

    def is_dark_cloud_cover(self, df):
        """Identifie un pattern Dark Cloud Cover"""
        if len(df) < 2:
            return False
            
        prev_candle = df.iloc[-2]
        curr_candle = df.iloc[-1]
        
        # Première bougie haussière
        if not (prev_candle['close'] > prev_candle['open']):
            return False
            
        # Deuxième bougie baissière
        if not (curr_candle['close'] < curr_candle['open']):
            return False
            
        # Ouverture en gap haussier
        if not (curr_candle['open'] > prev_candle['high']):
            return False
            
        # Clôture en-dessous du milieu de la première bougie
        prev_midpoint = (prev_candle['open'] + prev_candle['close']) / 2
        if not (curr_candle['close'] < prev_midpoint):
            return False
            
        return True

    def _cluster_levels(self, levels, threshold):
        """Regroupe les niveaux proches"""
        if not levels:
            return []
            
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) <= threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
                
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        return sorted(clusters)

class GPTTradingBot:
    def __init__(self):
        self.setup_logging()
        if not self.initialize_mt5():
            raise Exception("Échec de l'initialisation de MetaTrader 5")
        self.setup_telegram()
        self.daily_trades = 0
        self.daily_pl = 0
        self.last_reset = datetime.now().date()
        self.model = self.load_or_train_model()
        self.market_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.strategy_results = []
        self.symbols_data = {symbol: {} for symbol in SYMBOLS}
        self.positions = {}
        self.last_update = datetime.now()
        self.market_regime = None
        self.volatility_state = None
        self.sentiment_score = None
        self.performance_metrics = self.initialize_performance_metrics()
        self.pattern_recognizer = PatternRecognizer()
        self.trade_history = []
        self.economic_calendar = {}
        self.last_calendar_update = None
        self.calendar = ForexFactoryCalendar()
        self.news_analyzer = NewsAnalyzer()
        # Seuils de trading optimisés
        self.min_confidence_threshold = 80  # Réduit pour permettre plus d'opportunités
        self.min_signal_strength = 70       # Réduit mais compensé par des validations plus strictes
        self.validation_points_required = 8  # Augmenté pour une validation plus stricte
        self.min_trend_strength = 65        # Nouveau seuil pour la force de la tendance
        self.min_volume_threshold = 1.2     # Volume minimum par rapport à la moyenne
        self.spread_settings = {
            # Forex majeurs - spreads plus serrés
            'EURUSD': {'base': 1.2, 'volatility_factor': 0.2},
            'GBPUSD': {'base': 1.3, 'volatility_factor': 0.25},
            'USDJPY': {'base': 1.2, 'volatility_factor': 0.2},
            # Forex mineurs - spreads moyens
            'EURGBP': {'base': 1.4, 'volatility_factor': 0.3},
            'AUDUSD': {'base': 1.5, 'volatility_factor': 0.35},
            'USDCAD': {'base': 1.4, 'volatility_factor': 0.3},
            # Crypto - spreads plus larges
            'BTCUSD': {'base': 2.0, 'volatility_factor': 0.5},
            'ETHUSD': {'base': 2.0, 'volatility_factor': 0.5},
            # Valeurs par défaut pour les autres actifs
            'default': {'base': 1.5, 'volatility_factor': 0.4}
        }
        self.min_risk_reward = 2.0         # Ratio risque/récompense minimum

    def initialize_performance_metrics(self):
        """Initialise les métriques de performance du bot"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "risk_reward_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "daily_returns": [],
            "monthly_returns": [],
            "trades_by_symbol": {},
            "trades_by_session": {},
            "trades_by_strategy": {}
        }

    def analyze_with_ollama(self, market_data, symbol):
        """Version synchrone de l'analyse avec Ollama"""
        try:
            if isinstance(market_data, dict):
                if "H1" in market_data:
                    market_data = market_data["H1"]
                else:
                    logging.error(f"Données H1 non trouvées pour {symbol}")
                    return {
                        "decision": "HOLD",
                        "confidence": 0,
                        "market_context": {
                            "trend_strength": 0,
                            "volatility_rating": "LOW",
                            "trading_conditions": "UNFAVORABLE"
                        }
                    }

            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Impossible d'obtenir les informations du compte")
                return {
                    "decision": "HOLD",
                    "confidence": 0,
                    "market_context": {
                        "trend_strength": 0,
                        "volatility_rating": "LOW",
                        "trading_conditions": "UNFAVORABLE"
                    }
                }

            account_data = self.convert_to_json_serializable({
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "leverage": account_info.leverage,
                "daily_profit": self.daily_pl,
                "trades_today": self.daily_trades,
                "risk_per_trade": RISK_PER_TRADE_PERCENT
            })

            # Get base technical analysis
            base_technical = {
                "Trend": self.detect_trend(market_data),
                "Volatility": self.calculate_volatility(market_data)
            }

            # Technical analysis avancée
            technical_analysis = self.convert_to_json_serializable({
                "indicators": {
                    "RSI": {
                        "RSI14": float(market_data['RSI'].iloc[-1]),
                        "RSI5": float(market_data['RSI_5'].iloc[-1]),
                        "RSI21": float(market_data['RSI_21'].iloc[-1]),
                        "convergence": "BULLISH" if market_data['RSI_5'].iloc[-1] > market_data['RSI'].iloc[-1] > market_data['RSI_21'].iloc[-1] else
                                     "BEARISH" if market_data['RSI_5'].iloc[-1] < market_data['RSI'].iloc[-1] < market_data['RSI_21'].iloc[-1] else
                                     "NEUTRAL"
                    },
                    "MACD": {
                        "value": float(market_data['MACD'].iloc[-1]),
                        "signal": float(market_data['Signal'].iloc[-1]),
                        "histogram": float(market_data['MACD_Hist'].iloc[-1]),
                        "histogram_trend": "INCREASING" if market_data['MACD_Hist'].iloc[-1] > market_data['MACD_Hist'].iloc[-2] else "DECREASING"
                    },
                    "MA": {
                        "MA20": float(market_data['close'].rolling(20).mean().iloc[-1]),
                        "MA50": float(market_data['close'].rolling(50).mean().iloc[-1]),
                        "MA200": float(market_data['close'].rolling(200).mean().iloc[-1]),
                        "alignment": "BULLISH" if market_data['close'].rolling(20).mean().iloc[-1] > market_data['close'].rolling(50).mean().iloc[-1] > market_data['close'].rolling(200).mean().iloc[-1] else
                                   "BEARISH" if market_data['close'].rolling(20).mean().iloc[-1] < market_data['close'].rolling(50).mean().iloc[-1] < market_data['close'].rolling(200).mean().iloc[-1] else
                                   "MIXED"
                    },
                    "Bollinger": {
                        "standard": {
                            "upper": float(market_data['BB_upper'].iloc[-1]),
                            "middle": float(market_data['BB_middle'].iloc[-1]),
                            "lower": float(market_data['BB_lower'].iloc[-1])
                        },
                        "extended": {
                            "upper_3sd": float(market_data['BB_upper_3'].iloc[-1]),
                            "lower_3sd": float(market_data['BB_lower_3'].iloc[-1])
                        },
                        "width": float(market_data['BB_upper'].iloc[-1] - market_data['BB_lower'].iloc[-1]),
                        "width_trend": "EXPANDING" if (market_data['BB_upper'].iloc[-1] - market_data['BB_lower'].iloc[-1]) > (market_data['BB_upper'].iloc[-2] - market_data['BB_lower'].iloc[-2]) else "CONTRACTING"
                    },
                    "ATR": {
                        "ATR14": float(market_data['ATR'].iloc[-1]),
                        "ATR5": float(market_data['ATR_5'].iloc[-1]),
                        "ATR21": float(market_data['ATR_21'].iloc[-1]),
                        "trend": "INCREASING" if market_data['ATR_5'].iloc[-1] > market_data['ATR'].iloc[-1] > market_data['ATR_21'].iloc[-1] else
                                "DECREASING" if market_data['ATR_5'].iloc[-1] < market_data['ATR'].iloc[-1] < market_data['ATR_21'].iloc[-1] else
                                "STABLE"
                    },
                    "Momentum": {
                        "current": float(market_data['Momentum'].iloc[-1]),
                        "short_term": float(market_data['Momentum_5'].iloc[-1]),
                        "long_term": float(market_data['Momentum_21'].iloc[-1]),
                        "alignment": "BULLISH" if market_data['Momentum_5'].iloc[-1] > 0 and market_data['Momentum'].iloc[-1] > 0 and market_data['Momentum_21'].iloc[-1] > 0 else
                                   "BEARISH" if market_data['Momentum_5'].iloc[-1] < 0 and market_data['Momentum'].iloc[-1] < 0 and market_data['Momentum_21'].iloc[-1] < 0 else
                                   "MIXED"
                    },
                    "Volume": {
                        "VWAP": float(market_data['VWAP'].iloc[-1]),
                        "OBV": float(market_data['OBV'].iloc[-1]),
                        "OBV_MA": float(market_data['OBV_MA'].iloc[-1]),
                        "trend": "BULLISH" if market_data['OBV'].iloc[-1] > market_data['OBV_MA'].iloc[-1] else "BEARISH"
                    },
                    "Ichimoku": {
                        "Tenkan": float(market_data['Ichimoku_Tenkan'].iloc[-1]),
                        "Kijun": float(market_data['Ichimoku_Kijun'].iloc[-1]),
                        "Senkou_A": float(market_data['Ichimoku_Senkou_A'].iloc[-1]),
                        "Senkou_B": float(market_data['Ichimoku_Senkou_B'].iloc[-1]),
                        "Chikou": float(market_data['Ichimoku_Chikou'].iloc[-1]),
                        "cloud_status": "ABOVE_CLOUD" if market_data['close'].iloc[-1] > market_data['Ichimoku_Senkou_A'].iloc[-1] and market_data['close'].iloc[-1] > market_data['Ichimoku_Senkou_B'].iloc[-1] else
                                      "BELOW_CLOUD" if market_data['close'].iloc[-1] < market_data['Ichimoku_Senkou_A'].iloc[-1] and market_data['close'].iloc[-1] < market_data['Ichimoku_Senkou_B'].iloc[-1] else
                                      "IN_CLOUD"
                    },
                    "ADX": {
                        "ADX": float(market_data['ADX'].iloc[-1]),
                        "Plus_DI": float(market_data['Plus_DI'].iloc[-1]),
                        "Minus_DI": float(market_data['Minus_DI'].iloc[-1]),
                        "trend_strength": "STRONG" if market_data['ADX'].iloc[-1] > 25 else "WEAK",
                        "direction": "BULLISH" if market_data['Plus_DI'].iloc[-1] > market_data['Minus_DI'].iloc[-1] else "BEARISH"
                    }
                },
                "Trend": base_technical.get("Trend", "NEUTRAL"),
                "Volatility": base_technical.get("Volatility", 0)
            })

            # Patterns and key levels
            candlestick_patterns = self.pattern_recognizer.analyze_candle_patterns(market_data)
            key_levels = self.get_key_levels(market_data)
            volume_profile = self.analyze_volume_profile(market_data)

            # Market context avancé avec analyse approfondie
            market_context = self.convert_to_json_serializable({
                "trend_analysis": {
                    "strength": self.calculate_trend_strength(market_data),
                    "multi_timeframe": self.analyze_multi_timeframe_trend(symbol),
                    "momentum": {
                        "short_term": float(market_data['Momentum_5'].iloc[-1]),
                        "medium_term": float(market_data['Momentum'].iloc[-1]),
                        "long_term": float(market_data['Momentum_21'].iloc[-1]),
                        "alignment": "BULLISH" if market_data['Momentum_5'].iloc[-1] > 0 and market_data['Momentum'].iloc[-1] > 0 and market_data['Momentum_21'].iloc[-1] > 0 else
                                   "BEARISH" if market_data['Momentum_5'].iloc[-1] < 0 and market_data['Momentum'].iloc[-1] < 0 and market_data['Momentum_21'].iloc[-1] < 0 else
                                   "MIXED",
                        "strength_score": float(abs(market_data['Momentum'].iloc[-1]) / market_data['ATR'].iloc[-1] * 100)
                    },
                    "trend_quality": {
                        "consistency": "HIGH" if all(market_data['close'].diff().tail(5) > 0) or all(market_data['close'].diff().tail(5) < 0) else
                                     "MEDIUM" if abs(market_data['close'].diff().sum()) > market_data['ATR'].iloc[-1] * 3 else
                                     "LOW",
                        "pullbacks": "HEALTHY" if 0.3 <= (market_data['low'].iloc[-5:].min() - market_data['low'].min()) / (market_data['high'].max() - market_data['low'].min()) <= 0.7 else
                                    "EXTENDED"
                    }
                },
                "volatility_analysis": {
                    "current_state": self.volatility_state,
                    "atr_trend": {
                        "short_term": float(market_data['ATR_5'].iloc[-1]),
                        "medium_term": float(market_data['ATR'].iloc[-1]),
                        "long_term": float(market_data['ATR_21'].iloc[-1]),
                        "acceleration": (market_data['ATR_5'].iloc[-1] / market_data['ATR_21'].iloc[-1]) - 1,
                        "regime": "EXPANSION" if market_data['ATR_5'].iloc[-1] > market_data['ATR_21'].iloc[-1] * 1.1 else
                                 "CONTRACTION" if market_data['ATR_5'].iloc[-1] < market_data['ATR_21'].iloc[-1] * 0.9 else
                                 "STABLE"
                    },
                    "bollinger_state": {
                        "width": float(market_data['BB_upper'].iloc[-1] - market_data['BB_lower'].iloc[-1]),
                        "width_percentile": float(((market_data['BB_upper'] - market_data['BB_lower']) / market_data['close']).rank(pct=True).iloc[-1]),
                        "price_position": "ABOVE" if market_data['close'].iloc[-1] > market_data['BB_upper'].iloc[-1] else
                                        "BELOW" if market_data['close'].iloc[-1] < market_data['BB_lower'].iloc[-1] else
                                        "INSIDE",
                        "squeeze_status": "SQUEEZE" if (market_data['BB_upper'].iloc[-1] - market_data['BB_lower'].iloc[-1]) < 
                                                     (market_data['BB_upper'].iloc[-20:] - market_data['BB_lower'].iloc[-20:]).mean() * 0.85 else
                                        "EXPANSION" if (market_data['BB_upper'].iloc[-1] - market_data['BB_lower'].iloc[-1]) >
                                                     (market_data['BB_upper'].iloc[-20:] - market_data['BB_lower'].iloc[-20:]).mean() * 1.15 else
                                        "NORMAL"
                    },
                    "volatility_regime": {
                        "current": "HIGH" if market_data['ATR'].iloc[-1] > market_data['ATR'].rolling(20).mean().iloc[-1] * 1.5 else
                                  "LOW" if market_data['ATR'].iloc[-1] < market_data['ATR'].rolling(20).mean().iloc[-1] * 0.5 else
                                  "NORMAL",
                        "trend": "INCREASING" if market_data['ATR'].diff().rolling(5).mean().iloc[-1] > 0 else
                                "DECREASING" if market_data['ATR'].diff().rolling(5).mean().iloc[-1] < 0 else
                                "STABLE"
                    }
                },
                "market_regime": {
                    "current": self.detect_market_regime(market_data),
                    "volume_profile": volume_profile,
                    "volume_analysis": {
                        "relative_volume": float(market_data['tick_volume'].iloc[-1] / market_data['tick_volume'].rolling(20).mean().iloc[-1]),
                        "obv_trend": "BULLISH" if market_data['OBV'].iloc[-1] > market_data['OBV_MA'].iloc[-1] else "BEARISH",
                        "volume_trend": "INCREASING" if market_data['tick_volume'].diff().mean() > 0 else "DECREASING",
                        "volume_quality": {
                            "trend_confirmation": "CONFIRMED" if (market_data['close'].diff().iloc[-1] > 0 and market_data['tick_volume'].diff().iloc[-1] > 0) or
                                                               (market_data['close'].diff().iloc[-1] < 0 and market_data['tick_volume'].diff().iloc[-1] > 0) else
                                                "DIVERGENT",
                            "consistency": "HIGH" if market_data['tick_volume'].iloc[-1] > market_data['tick_volume'].rolling(20).mean().iloc[-1] * 1.5 else
                                         "LOW" if market_data['tick_volume'].iloc[-1] < market_data['tick_volume'].rolling(20).mean().iloc[-1] * 0.5 else
                                         "NORMAL"
                        }
                    },
                    "market_phase": {
                        "current": self._detect_market_phase(market_data),
                        "strength": float(abs(market_data['close'].diff().rolling(5).mean().iloc[-1]) / market_data['ATR'].iloc[-1] * 100),
                        "characteristics": {
                            "volume_participation": "HIGH" if market_data['tick_volume'].iloc[-1] > market_data['tick_volume'].rolling(20).mean().iloc[-1] * 1.5 else
                                                 "LOW" if market_data['tick_volume'].iloc[-1] < market_data['tick_volume'].rolling(20).mean().iloc[-1] * 0.5 else
                                                 "NORMAL",
                            "price_action": "IMPULSIVE" if abs(market_data['close'].diff().iloc[-1]) > market_data['ATR'].iloc[-1] else
                                          "CORRECTIVE" if abs(market_data['close'].diff().iloc[-1]) < market_data['ATR'].iloc[-1] * 0.3 else
                                          "NORMAL",
                            "momentum_quality": "STRONG" if abs(market_data['RSI'].iloc[-1] - 50) > 20 else
                                              "WEAK" if abs(market_data['RSI'].iloc[-1] - 50) < 10 else
                                              "MODERATE"
                        },
                        "confirmation": {
                            "volume_confirms_price": "YES" if (market_data['close'].diff().iloc[-1] > 0 and market_data['tick_volume'].diff().iloc[-1] > 0) or
                                                            (market_data['close'].diff().iloc[-1] < 0 and market_data['tick_volume'].diff().iloc[-1] > 0) else
                                                   "NO",
                            "momentum_confirms_trend": "YES" if (market_data['close'].diff().iloc[-1] > 0 and market_data['RSI'].diff().iloc[-1] > 0) or
                                                              (market_data['close'].diff().iloc[-1] < 0 and market_data['RSI'].diff().iloc[-1] < 0) else
                                                     "NO",
                            "structure_confirms_phase": "YES" if self._validate_market_structure(market_data) else "NO"
                        }
                    }
                },
                "support_resistance": {
                    "dynamic": self.calculate_dynamic_sr_levels(market_data),
                    "static": self.get_key_levels(market_data),
                    "fibonacci": self.calculate_fibonacci_levels(market_data),
                    "pivot_points": self.calculate_pivot_points(market_data)
                },
                "market_structure": {
                    "key_levels": key_levels,
                    "ichimoku_analysis": {
                        "cloud_status": "ABOVE_CLOUD" if market_data['close'].iloc[-1] > market_data['Ichimoku_Senkou_A'].iloc[-1] and 
                                                       market_data['close'].iloc[-1] > market_data['Ichimoku_Senkou_B'].iloc[-1] else
                                      "BELOW_CLOUD" if market_data['close'].iloc[-1] < market_data['Ichimoku_Senkou_A'].iloc[-1] and 
                                                     market_data['close'].iloc[-1] < market_data['Ichimoku_Senkou_B'].iloc[-1] else
                                      "IN_CLOUD",
                        "trend_strength": "STRONG" if market_data['Ichimoku_Chikou'].iloc[-1] > market_data['close'].iloc[-1] else "WEAK",
                        "future_sentiment": "BULLISH" if market_data['Ichimoku_Senkou_A'].iloc[-1] > market_data['Ichimoku_Senkou_B'].iloc[-1] else "BEARISH"
                    },
                    "price_action": {
                        "trend_structure": "HIGHER_HIGHS" if all(market_data['high'].diff().tail(3) > 0) else
                                         "LOWER_LOWS" if all(market_data['low'].diff().tail(3) < 0) else
                                         "SIDEWAYS",
                        "swing_magnitude": float(abs(market_data['high'].max() - market_data['low'].min()) / market_data['close'].mean()),
                        "momentum_divergence": "BULLISH" if market_data['RSI'].iloc[-1] > market_data['RSI'].iloc[-2] and market_data['close'].iloc[-1] < market_data['close'].iloc[-2] else
                                             "BEARISH" if market_data['RSI'].iloc[-1] < market_data['RSI'].iloc[-2] and market_data['close'].iloc[-1] > market_data['close'].iloc[-2] else
                                             "NONE"
                    }
                }
            })

            # Performance data
            performance_data = self.get_recent_performance(symbol)
            recent_trades = self.convert_to_json_serializable(self.trade_history[-10:])

            # Economic events and news
            economic_events = self.calendar.get_events_for_symbol(symbol) if self.calendar else {}
            sentiment_score, confidence_score = self.news_analyzer.get_market_sentiment(symbol)
            news_impact = {
                "sentiment_score": sentiment_score,
                "confidence_score": confidence_score
            }

            # Latest prices
            latest_prices = self.convert_to_json_serializable({
                "current": float(market_data['close'].iloc[-1]),
                "previous": float(market_data['close'].iloc[-2]),
                "high_24h": float(market_data['high'].max()),
                "low_24h": float(market_data['low'].min()),
                "volume_24h": float(market_data['tick_volume'].sum()),
                "price_change_24h": float((market_data['close'].iloc[-1] - market_data['close'].iloc[-24]) / market_data['close'].iloc[-24] * 100),
                "spread": float(mt5.symbol_info(symbol).spread) if mt5.symbol_info(symbol) else 0
            })

            # Format prompt
            prompt = f"""Analyze this comprehensive market data for {symbol}:

Technical Analysis:
{json.dumps(technical_analysis, indent=2)}

Market Context:
{json.dumps(market_context, indent=2)}

Account Information:
{json.dumps(account_data, indent=2)}

Latest Price Data:
{json.dumps(latest_prices, indent=2)}

Patterns Found:
{[str(pattern) for pattern in candlestick_patterns]}

Economic Events:
{json.dumps(economic_events, indent=2)}

Performance Metrics:
{json.dumps(performance_data, indent=2)}

News Sentiment:
{json.dumps(news_impact, indent=2)}

Please analyze all the above data and provide a detailed trading decision with advanced risk management in this exact JSON format:
{{
    "decision": "BUY|SELL|HOLD",
    "confidence": <0-100>,
    "recommended_position_size": <float>,
    "entry": {{
        "price": <float>,
        "type": "MARKET|LIMIT",
        "rationale": <string>
    }},
    "risk_management": {{
        "stop_loss": <float>,
        "take_profit": <float>,
        "position_scale_levels": [<float array>]
    }},
    "analysis": {{
        "primary_factors": [<string array>],
        "supporting_factors": [<string array>],
        "risk_factors": [<string array>],
        "market_structure": {{
            "trend": <string>,
            "strength": <0-100>,
            "key_levels": {{
                "support": <float array>,
                "resistance": <float array>
            }}
        }}
    }},
    "market_context": {{
        "trend_strength": <0-100>,
        "volatility_rating": "LOW|MEDIUM|HIGH",
        "trading_conditions": "FAVORABLE|NEUTRAL|UNFAVORABLE",
        "risk_level": "LOW|MEDIUM|HIGH"
    }}
}}"""

            print("\n\033[94m[Deepseek Analysis Request]\033[0m")
            print(f"\033[93mAnalyzing {symbol}...\033[0m")

            # Call Ollama API
            response = post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "deepseek-r1:14b",
                    "prompt": prompt,
                    "stream": True
                },
                stream=True
            )

            full_response = ""
            print("\n\033[92m[Deepseek Response]\033[0m")
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line.decode('utf-8'))
                    if 'response' in json_response:
                        chunk = json_response['response']
                        full_response += chunk
                        print(chunk, end='', flush=True)

            print("\n")

            # Parse response
            try:
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    analysis_result = json.loads(full_response[json_start:json_end])
                    print("\n\033[96m[Parsed Analysis]\033[0m")
                    print(json.dumps(analysis_result, indent=2))
                    return analysis_result
            except json.JSONDecodeError as e:
                logging.error(f"Erreur de parsing JSON: {e}")
                return self.parse_gpt_response(full_response)

            return {
                "decision": "HOLD",
                "confidence": 0,
                "market_context": {
                    "trend_strength": 0,
                    "volatility_rating": "LOW",
                    "trading_conditions": "UNFAVORABLE"
                }
            }

        except Exception as e:
            logging.error(f"Erreur dans l'analyse Ollama: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "market_context": {
                    "trend_strength": 0,
                    "volatility_rating": "LOW",
                    "trading_conditions": "UNFAVORABLE"
                }
            }

    def analyze_with_gpt4(self, market_data, symbol):
        """Version synchrone de l'analyse avec OpenRouter"""
        try:
            if isinstance(market_data, dict):
                if "H1" in market_data:
                    market_data = market_data["H1"]
                else:
                    logging.error(f"Données H1 non trouvées pour {symbol}")
                    return {
                        "decision": "HOLD",
                        "confidence": 0,
                        "market_context": {
                            "trend_strength": 0,
                            "volatility_rating": "LOW",
                            "trading_conditions": "UNFAVORABLE"
                        }
                    }

            # Préparation des données
            account_info = mt5.account_info()
            if account_info is None:
                logging.error("Impossible d'obtenir les informations du compte")
                return {
                    "decision": "HOLD",
                    "confidence": 0,
                    "market_context": {
                        "trend_strength": 0,
                        "volatility_rating": "LOW",
                        "trading_conditions": "UNFAVORABLE"
                    }
                }

            # Préparation des données pour l'analyse
            technical_analysis = self.convert_to_json_serializable({
                "indicators": {
                    "RSI": {
                        "RSI14": float(market_data['RSI'].iloc[-1]),
                        "RSI5": float(market_data['RSI_5'].iloc[-1]),
                        "RSI21": float(market_data['RSI_21'].iloc[-1])
                    },
                    "MACD": {
                        "value": float(market_data['MACD'].iloc[-1]),
                        "signal": float(market_data['Signal'].iloc[-1]),
                        "histogram": float(market_data['MACD_Hist'].iloc[-1])
                    }
                }
            })

            # Préparation du prompt
            prompt = f"""Analyze this market data for {symbol}:

Technical Analysis:
{json.dumps(technical_analysis, indent=2)}

Please analyze the data and provide a trading decision in this JSON format:
{{
    "decision": "BUY|SELL|HOLD",
    "confidence": <0-100>,
    "market_context": {{
        "trend_strength": <0-100>,
        "volatility_rating": "LOW|MEDIUM|HIGH",
        "trading_conditions": "FAVORABLE|NEUTRAL|UNFAVORABLE"
    }}
}}"""

            # Appel à l'API OpenRouter
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "Trading Bot"
            }

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

            if response.status_code != 200:
                logging.error(f"Erreur OpenRouter: {response.text}")
                return {
                    "decision": "HOLD",
                    "confidence": 0,
                    "market_context": {
                        "trend_strength": 0,
                        "volatility_rating": "LOW",
                        "trading_conditions": "UNFAVORABLE"
                    }
                }

            # Traitement de la réponse
            try:
                response_data = response.json()
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    content = response_data["choices"][0]["message"]["content"]
                    # Extraire le JSON de la réponse
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > 0:
                        analysis_result = json.loads(content[json_start:json_end])
                        return analysis_result
                    else:
                        return self.parse_gpt_response(content)
                else:
                    logging.error("Réponse OpenRouter invalide")
                    return {
                        "decision": "HOLD",
                        "confidence": 0,
                        "market_context": {
                            "trend_strength": 0,
                            "volatility_rating": "LOW",
                            "trading_conditions": "UNFAVORABLE"
                        }
                    }

            except json.JSONDecodeError as e:
                logging.error(f"Erreur de parsing JSON: {e}")
                return self.parse_gpt_response(content)

        except Exception as e:
            logging.error(f"Erreur dans l'analyse OpenRouter: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "market_context": {
                    "trend_strength": 0,
                    "volatility_rating": "LOW",
                    "trading_conditions": "UNFAVORABLE"
                }
            }

    def run(self):
        """Boucle principale d'exécution du bot"""
        try:
            print("Bot démarré... Appuyez sur Ctrl+C pour arrêter")
            logging.info("Bot démarré")
            
            while True:
                try:
                    if not self.check_market_conditions():
                        time_to_next = self.get_time_to_next_session()
                        logging.info(f"Marché fermé. Prochaine session dans {time_to_next} secondes")
                        time.sleep(min(time_to_next, 600))
                        continue

                    logging.info("Début du cycle de scan des marchés")
                    for symbol in SYMBOLS:
                        try:
                            logging.info(f"Analyse de {symbol}...")
                            market_data = self.get_market_data_multi_timeframe(symbol)
                            
                            if market_data is None:
                                logging.error(f"Impossible de récupérer les données pour {symbol}")
                                continue
                                
                            if "H1" in market_data and market_data["H1"] is not None and not market_data["H1"].empty:
                                analysis = self.analyze_with_gpt4(market_data["H1"], symbol)
                                score = self.calculate_market_score(market_data["H1"].iloc[-1])
                                
                                if analysis["confidence"] >= 75 and score >= MIN_SCORE_TO_TRADE:
                                    if self.validate_trade_conditions(analysis, symbol):
                                        self.place_trade(analysis["decision"], symbol, 
                                                       suggested_size=analysis.get("recommended_position_size"))
                                        logging.info(f"Signal de trading validé pour {symbol}")
                                    else:
                                        logging.info(f"Conditions non validées pour {symbol}")
                                else:
                                    logging.info(f"Score ou confiance insuffisant pour {symbol}: {score}/{MIN_SCORE_TO_TRADE}, {analysis['confidence']}/75")

                            time.sleep(5)

                        except Exception as e:
                            logging.error(f"Erreur lors de l'analyse de {symbol}: {e}")
                            continue

                    self.manage_positions()
                    logging.info("Scan terminé, attente du prochain cycle dans 30 minutes")
                    time.sleep(1800)

                except Exception as e:
                    logging.error(f"Erreur dans la boucle principale: {e}")
                    time.sleep(60)

        except KeyboardInterrupt:
            logging.info("Arrêt du bot par l'utilisateur")
            mt5.shutdown()
        except Exception as e:
            logging.error(f"Erreur fatale: {e}")
            mt5.shutdown()
            raise

    def get_market_data_multi_timeframe(self, symbol):
        """Récupère les données de marché pour différentes timeframes"""
        try:
            timeframes = {
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            result = {}
            for tf_name, tf_value in timeframes.items():
                # Récupérer les données
                rates = mt5.copy_rates_from(symbol, tf_value, datetime.now(), 1000)
                if rates is None or len(rates) == 0:
                    logging.error(f"Impossible de récupérer les données pour {symbol} sur {tf_name}")
                    continue
                    
                # Convertir en DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Calculer les indicateurs
                df = self.calculate_indicators(df)
                
                result[tf_name] = df
                
            return result
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des données multi-timeframes pour {symbol}: {e}")
            return None
            
    def calculate_indicators(self, df):
        """Calcule les indicateurs techniques avancés"""
        try:
            # RSI avec période personnalisable
            def calculate_rsi(data, period=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            # RSI sur plusieurs périodes
            df['RSI'] = calculate_rsi(df['close'], 14)
            df['RSI_21'] = calculate_rsi(df['close'], 21)
            df['RSI_5'] = calculate_rsi(df['close'], 5)
            
            # MACD avec paramètres personnalisables
            def calculate_macd(data, fast=12, slow=26, signal=9):
                exp1 = data.ewm(span=fast, adjust=False).mean()
                exp2 = data.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                return macd, signal_line

            df['MACD'], df['Signal'] = calculate_macd(df['close'])
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            # Bollinger Bands avec déviations multiples
            def calculate_bollinger_bands(data, period=20, std_dev=2):
                middle = data.rolling(window=period).mean()
                std = data.rolling(window=period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                return middle, upper, lower

            df['BB_middle'], df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['close'])
            # Bandes supplémentaires pour plus de précision
            _, df['BB_upper_3'], df['BB_lower_3'] = calculate_bollinger_bands(df['close'], std_dev=3)
            
            # ATR amélioré avec plusieurs périodes
            def calculate_atr(data, period=14):
                high_low = data['high'] - data['low']
                high_close = np.abs(data['high'] - data['close'].shift())
                low_close = np.abs(data['low'] - data['close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                return true_range.rolling(period).mean()

            df['ATR'] = calculate_atr(df)
            df['ATR_5'] = calculate_atr(df, 5)
            df['ATR_21'] = calculate_atr(df, 21)
            
            # Momentum sur plusieurs périodes
            df['Momentum'] = df['close'] - df['close'].shift(10)
            df['Momentum_5'] = df['close'] - df['close'].shift(5)
            df['Momentum_21'] = df['close'] - df['close'].shift(21)
            
            # VWAP amélioré avec reset quotidien
            def calculate_vwap(data):
                data = data.copy()
                data['Date'] = pd.to_datetime(data['time']).dt.date
                vwap = (data['close'] * data['tick_volume']).groupby(data['Date']).cumsum() / \
                       data['tick_volume'].groupby(data['Date']).cumsum()
                return vwap

            df['VWAP'] = calculate_vwap(df)
            
            # OBV avec moyenne mobile
            df['OBV'] = (np.sign(df['close'].diff()) * df['tick_volume']).fillna(0).cumsum()
            df['OBV_MA'] = df['OBV'].rolling(20).mean()
            
            # Stochastic RSI
            def calculate_stoch_rsi(data, period=14, smoothk=3, smoothd=3):
                rsi = calculate_rsi(data, period)
                stoch_rsi = (rsi - rsi.rolling(period).min()) / \
                           (rsi.rolling(period).max() - rsi.rolling(period).min())
                k = stoch_rsi.rolling(smoothk).mean()
                d = k.rolling(smoothd).mean()
                return k, d

            df['Stoch_RSI_K'], df['Stoch_RSI_D'] = calculate_stoch_rsi(df['close'])
            
            # Ichimoku Cloud
            def calculate_ichimoku(data):
                high_9 = data['high'].rolling(9).max()
                low_9 = data['low'].rolling(9).min()
                tenkan_sen = (high_9 + low_9) / 2
                
                high_26 = data['high'].rolling(26).max()
                low_26 = data['low'].rolling(26).min()
                kijun_sen = (high_26 + low_26) / 2
                
                senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
                
                high_52 = data['high'].rolling(52).max()
                low_52 = data['low'].rolling(52).min()
                senkou_span_b = ((high_52 + low_52) / 2).shift(26)
                
                chikou_span = data['close'].shift(-26)
                
                return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

            (df['Ichimoku_Tenkan'], df['Ichimoku_Kijun'],
             df['Ichimoku_Senkou_A'], df['Ichimoku_Senkou_B'],
             df['Ichimoku_Chikou']) = calculate_ichimoku(df)
            
            # ADX (Average Directional Index)
            def calculate_adx(data, period=14):
                plus_dm = data['high'].diff()
                minus_dm = data['low'].diff()
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm > 0] = 0
                
                tr1 = pd.DataFrame(data['high'] - data['low'])
                tr2 = pd.DataFrame(abs(data['high'] - data['close'].shift(1)))
                tr3 = pd.DataFrame(abs(data['low'] - data['close'].shift(1)))
                frames = [tr1, tr2, tr3]
                tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
                atr = tr.rolling(period).mean()
                
                plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
                minus_di = abs(100 * (minus_dm.rolling(period).mean() / atr))
                dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
                adx = dx.rolling(period).mean()
                
                return adx, plus_di, minus_di

            df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des indicateurs: {e}")
            return df

    def detect_trend(self, df):
        """Détecte la tendance du marché en utilisant une analyse multi-indicateurs"""
        try:
            trend_signals = []
            
            # 1. Analyse des moyennes mobiles (25%)
            ma20 = df['close'].rolling(window=20).mean()
            ma50 = df['close'].rolling(window=50).mean()
            ma200 = df['close'].rolling(window=200).mean()
            
            ma_score = 0
            if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
                ma_score = 1
            elif ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
                ma_score = -1
            trend_signals.append(ma_score)
            
            # 2. Analyse ADX et DI (25%)
            adx = df['ADX'].iloc[-1]
            plus_di = df['Plus_DI'].iloc[-1]
            minus_di = df['Minus_DI'].iloc[-1]
            
            adx_score = 0
            if adx > 25:  # Tendance significative
                if plus_di > minus_di:
                    adx_score = 1
                elif minus_di > plus_di:
                    adx_score = -1
            trend_signals.append(adx_score)
            
            # 3. Analyse Ichimoku (25%)
            ichimoku_score = 0
            price = df['close'].iloc[-1]
            if (price > df['Ichimoku_Senkou_A'].iloc[-1] and 
                price > df['Ichimoku_Senkou_B'].iloc[-1] and
                df['Ichimoku_Tenkan'].iloc[-1] > df['Ichimoku_Kijun'].iloc[-1]):
                ichimoku_score = 1
            elif (price < df['Ichimoku_Senkou_A'].iloc[-1] and 
                  price < df['Ichimoku_Senkou_B'].iloc[-1] and
                  df['Ichimoku_Tenkan'].iloc[-1] < df['Ichimoku_Kijun'].iloc[-1]):
                ichimoku_score = -1
            trend_signals.append(ichimoku_score)
            
            # 4. Analyse MACD et RSI (25%)
            macd_signal = 0
            if df['MACD'].iloc[-1] > df['Signal'].iloc[-1]:
                if df['MACD_Hist'].iloc[-1] > df['MACD_Hist'].iloc[-2]:
                    macd_signal = 1
            elif df['MACD'].iloc[-1] < df['Signal'].iloc[-1]:
                if df['MACD_Hist'].iloc[-1] < df['MACD_Hist'].iloc[-2]:
                    macd_signal = -1
                    
            rsi = df['RSI'].iloc[-1]
            rsi_signal = 0
            if rsi > 60:  # Zone de surachat modérée
                rsi_signal = 1
            elif rsi < 40:  # Zone de survente modérée
                rsi_signal = -1
                
            momentum_score = (macd_signal + rsi_signal) / 2
            trend_signals.append(momentum_score)
            
            # Calculer le score final
            final_score = sum(trend_signals) / len(trend_signals)
            
            # Déterminer la force de la tendance
            trend_strength = abs(final_score)
            
            # Classifier la tendance
            if trend_strength >= 0.7:
                if final_score > 0:
                    return "STRONG_UPTREND"
                else:
                    return "STRONG_DOWNTREND"
            elif trend_strength >= 0.3:
                if final_score > 0:
                    return "UPTREND"
                else:
                    return "DOWNTREND"
            else:
                # Vérifier les conditions de range
                bb_width = df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]
                avg_bb_width = (df['BB_upper'] - df['BB_lower']).mean()
                
                if bb_width < avg_bb_width * 0.8:
                    return "RANGING_TIGHT"
                elif bb_width > avg_bb_width * 1.2:
                    return "RANGING_WIDE"
                else:
                    return "RANGING_NORMAL"
                
        except Exception as e:
            logging.error(f"Erreur lors de la détection de tendance: {e}")
            return "NEUTRAL"
            
    def calculate_volatility(self, df):
        """Calcule la volatilité du marché"""
        try:
            # Utiliser l'ATR comme mesure de volatilité
            atr = df['ATR'].iloc[-1]
            avg_price = df['close'].mean()
            volatility = (atr / avg_price) * 100
            
            # Normaliser entre 0 et 100
            return min(max(volatility * 10, 0), 100)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul de la volatilité: {e}")
            return 0

    def analyze_multi_timeframe_trend(self, symbol):
        """Analyse la tendance sur plusieurs timeframes"""
        try:
            trends = {}
            for tf, data in self.market_cache.get(symbol, {}).items():
                if data is not None and not data.empty:
                    trends[tf] = self.detect_trend(data)
            return trends
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse multi-timeframe: {e}")
            return {}

    def calculate_trend_strength(self, df):
        """Calcule la force de la tendance"""
        try:
            # Utiliser RSI, ADX et la pente des MAs
            rsi = df['RSI'].iloc[-1]
            ma_slopes = {
                'MA20': (df['close'].rolling(20).mean().diff() / df['close'].rolling(20).mean()).iloc[-1],
                'MA50': (df['close'].rolling(50).mean().diff() / df['close'].rolling(50).mean()).iloc[-1]
            }
            
            # Normaliser entre 0 et 100
            strength = (abs(rsi - 50) * 2 + 
                      abs(ma_slopes['MA20']) * 1000 + 
                      abs(ma_slopes['MA50']) * 1000) / 3
                      
            return min(max(strength, 0), 100)
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul de la force de tendance: {e}")
            return 0

    def _detect_market_phase(self, df):
        """Détecte la phase de marché actuelle avec une analyse approfondie"""
        try:
            # 1. Analyse des prix et volumes
            price = df['close'].iloc[-1]
            price_sma = df['close'].rolling(20).mean().iloc[-1]
            volume_trend = df['tick_volume'].diff().rolling(5).mean().iloc[-1]
            obv_trend = df['OBV'].diff().rolling(5).mean().iloc[-1]
            
            # 2. Analyse de la structure de marché
            highs = df['high'].rolling(5).max()
            lows = df['low'].rolling(5).min()
            higher_highs = (highs.diff() > 0).rolling(3).sum().iloc[-1]
            lower_lows = (lows.diff() < 0).rolling(3).sum().iloc[-1]
            
            # 3. Analyse de la volatilité et du momentum
            volatility = self.calculate_volatility(df)
            rsi = df['RSI'].iloc[-1]
            momentum = df['Momentum'].iloc[-1]
            
            # Détection de la phase
            if (price < df['BB_lower'].iloc[-1] and 
                obv_trend > 0 and 
                volume_trend > 0 and 
                rsi < 40):
                # Phase d'accumulation
                return "ACCUMULATION"
                
            elif (price > df['BB_upper'].iloc[-1] and 
                  obv_trend < 0 and 
                  volume_trend > 0 and 
                  rsi > 60):
                # Phase de distribution
                return "DISTRIBUTION"
                
            elif (higher_highs >= 2 and 
                  momentum > 0 and 
                  price > price_sma and 
                  volume_trend > 0):
                # Phase de markup
                return "MARKUP"
                
            elif (lower_lows >= 2 and 
                  momentum < 0 and 
                  price < price_sma and 
                  volume_trend > 0):
                # Phase de markdown
                return "MARKDOWN"
                
            else:
                # Phase de consolidation
                return "CONSOLIDATION"
                
        except Exception as e:
            logging.error(f"Erreur lors de la détection de la phase de marché: {e}")
            return "CONSOLIDATION"
            
    def _validate_market_structure(self, df):
        """Valide la structure de marché pour confirmer la phase"""
        try:
            # Analyser les dernières 20 bougies
            recent_data = df.tail(20)
            
            # 1. Vérifier la cohérence des prix
            price_trend = recent_data['close'].diff().rolling(5).mean().iloc[-1]
            volume_trend = recent_data['tick_volume'].diff().rolling(5).mean().iloc[-1]
            
            # 2. Vérifier les pivots
            highs = recent_data[recent_data['high'] == recent_data['high'].rolling(5).max()]
            lows = recent_data[recent_data['low'] == recent_data['low'].rolling(5).min()]
            
            # 3. Vérifier le momentum
            momentum_consistent = (
                (price_trend > 0 and recent_data['RSI'].diff().mean() > 0) or
                (price_trend < 0 and recent_data['RSI'].diff().mean() < 0)
            )
            
            # 4. Vérifier la structure des volumes
            volume_consistent = (
                volume_trend > 0 and
                recent_data['tick_volume'].iloc[-1] > recent_data['tick_volume'].mean()
            )
            
            # 5. Vérifier les supports/résistances
            sr_respected = (
                recent_data['close'].iloc[-1] > recent_data['BB_lower'].iloc[-1] and
                recent_data['close'].iloc[-1] < recent_data['BB_upper'].iloc[-1]
            )
            
            # Validation finale
            return (
                momentum_consistent and
                volume_consistent and
                sr_respected and
                len(highs) + len(lows) >= 3  # Au moins 3 pivots identifiés
            )
            
        except Exception as e:
            logging.error(f"Erreur lors de la validation de la structure de marché: {e}")
            return False

    def detect_market_regime(self, df):
        """Détecte le régime de marché avec analyse multi-facteurs"""
        try:
            # 1. Analyse de la volatilité (25%)
            volatility = self.calculate_volatility(df)
            atr_acceleration = (df['ATR_5'].iloc[-1] / df['ATR_21'].iloc[-1]) - 1
            
            volatility_score = 0
            if volatility > 70:
                volatility_score = 2  # Haute volatilité
            elif volatility > 40:
                volatility_score = 1  # Volatilité moyenne
                
            if atr_acceleration > 0.1:
                volatility_score += 1  # Accélération de la volatilité
            
            # 2. Analyse de la tendance (25%)
            trend = self.detect_trend(df)
            trend_score = 0
            
            if trend in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                trend_score = 2
            elif trend in ["UPTREND", "DOWNTREND"]:
                trend_score = 1
            
            # 3. Analyse du volume (25%)
            volume = df['tick_volume'].iloc[-20:]
            avg_volume = volume.mean()
            recent_volume = volume.iloc[-1]
            volume_trend = volume.diff().mean()
            
            volume_score = 0
            if recent_volume > avg_volume * 1.5:
                volume_score += 1
            if volume_trend > 0:
                volume_score += 1
            
            # 4. Analyse de la structure de marché (25%)
            bb_width = df['BB_upper'].iloc[-1] - df['BB_lower'].iloc[-1]
            avg_bb_width = (df['BB_upper'] - df['BB_lower']).mean()
            price = df['close'].iloc[-1]
            
            structure_score = 0
            # Expansion/Contraction des bandes
            if bb_width > avg_bb_width * 1.2:
                structure_score += 1
            elif bb_width < avg_bb_width * 0.8:
                structure_score -= 1
                
            # Position par rapport aux bandes
            if price > df['BB_upper'].iloc[-1] or price < df['BB_lower'].iloc[-1]:
                structure_score += 1
            
            # Calculer le score total
            total_score = (volatility_score + trend_score + volume_score + structure_score) / 4
            
            # Classifier le régime
            if trend_score >= 1.5:  # Forte tendance
                if volatility_score >= 1.5:
                    return "TRENDING_VOLATILE"
                return "TRENDING_STABLE"
            elif volatility_score >= 1.5:  # Haute volatilité sans tendance claire
                if volume_score >= 1.5:
                    return "RANGING_VOLATILE_HIGH_VOLUME"
                return "RANGING_VOLATILE"
            elif structure_score <= -0.5:  # Compression
                return "RANGING_COMPRESSION"
            elif structure_score >= 1:  # Expansion
                return "RANGING_EXPANSION"
            elif volume_score <= 0.5:  # Faible activité
                return "RANGING_LOW_VOLUME"
            else:
                return "RANGING_NORMAL"
            
        except Exception as e:
            logging.error(f"Erreur lors de la détection du régime de marché: {e}")
            return "RANGING_STABLE"

    def analyze_volume_profile(self, df):
        """Analyse le profil de volume"""
        try:
            # Calculer la distribution du volume par niveau de prix
            price_levels = pd.qcut(df['close'], q=10)
            volume_profile = df.groupby(price_levels)['tick_volume'].sum()
            
            # Identifier les niveaux de prix à fort volume
            high_volume_levels = volume_profile[volume_profile > volume_profile.mean()]
            
            # Convertir les Interval objects en strings pour la sérialisation JSON
            volume_distribution = {str(k): v for k, v in volume_profile.to_dict().items()}
            high_volume_list = [float(x.mid) for x in high_volume_levels.index]
            
            return {
                "high_volume_levels": high_volume_list,
                "volume_distribution": volume_distribution
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse du profil de volume: {e}")
            return {"high_volume_levels": [], "volume_distribution": {}}

    def calculate_fibonacci_levels(self, df):
        """Calcule les niveaux de Fibonacci"""
        try:
            high = df['high'].max()
            low = df['low'].min()
            diff = high - low
            
            return {
                "0": low,
                "0.236": low + 0.236 * diff,
                "0.382": low + 0.382 * diff,
                "0.5": low + 0.5 * diff,
                "0.618": low + 0.618 * diff,
                "0.786": low + 0.786 * diff,
                "1": high
            }
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des niveaux Fibonacci: {e}")
            return {}

    def calculate_pivot_points(self, df):
        """Calcule les points pivots"""
        try:
            last_day = df.iloc[-1]
            high = last_day['high']
            low = last_day['low']
            close = last_day['close']
            
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            
            return {
                "pivot": pivot,
                "r1": r1,
                "r2": r2,
                "s1": s1,
                "s2": s2
            }
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des points pivots: {e}")
            return {}

    def calculate_dynamic_sr_levels(self, df):
        """Calcule les niveaux de support/résistance dynamiques"""
        try:
            # Utiliser les bandes de Bollinger comme niveaux dynamiques
            return {
                "upper": float(df['BB_upper'].iloc[-1]),
                "middle": float(df['BB_middle'].iloc[-1]),
                "lower": float(df['BB_lower'].iloc[-1])
            }
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul des niveaux S/R dynamiques: {e}")
            return {}

    def _cluster_levels(self, levels, threshold):
        """Regroupe les niveaux proches"""
        if not levels:
            return []
            
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) <= threshold:
                current_cluster.append(level)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
                
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        return sorted(clusters)

    def get_key_levels(self, df):
        """Identifie les niveaux clés du marché"""
        try:
            # Identifier les sommets et creux locaux
            highs = df[df['high'] == df['high'].rolling(5).max()]
            lows = df[df['low'] == df['low'].rolling(5).min()]
            
            # Regrouper les niveaux proches
            resistance_levels = self._cluster_levels(highs['high'].tolist(), df['ATR'].mean())
            support_levels = self._cluster_levels(lows['low'].tolist(), df['ATR'].mean())
            
            return {
                "resistance": resistance_levels[-5:] if len(resistance_levels) > 5 else resistance_levels,
                "support": support_levels[-5:] if len(support_levels) > 5 else support_levels
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'identification des niveaux clés: {e}")
            return {"resistance": [], "support": []}

    def parse_gpt_response(self, response_text):
        """Parse la réponse GPT en cas d'échec du parsing JSON"""
        try:
            # Valeurs par défaut
            result = {
                "decision": "HOLD",
                "confidence": 0,
                "market_context": {
                    "trend_strength": 0,
                    "volatility_rating": "LOW",
                    "trading_conditions": "UNFAVORABLE"
                }
            }
            
            # Rechercher des patterns clés dans la réponse
            if "BUY" in response_text.upper():
                result["decision"] = "BUY"
            elif "SELL" in response_text.upper():
                result["decision"] = "SELL"
                
            # Rechercher un niveau de confiance
            confidence_match = re.search(r"confidence.*?(\d+)", response_text, re.IGNORECASE)
            if confidence_match:
                result["confidence"] = int(confidence_match.group(1))
                
            # Rechercher la force de la tendance
            trend_match = re.search(r"trend.*?strength.*?(\d+)", response_text, re.IGNORECASE)
            if trend_match:
                result["market_context"]["trend_strength"] = int(trend_match.group(1))
                
            # Rechercher la volatilité
            if "HIGH VOLATILITY" in response_text.upper():
                result["market_context"]["volatility_rating"] = "HIGH"
            elif "MEDIUM VOLATILITY" in response_text.upper():
                result["market_context"]["volatility_rating"] = "MEDIUM"
                
            # Rechercher les conditions de trading
            if "FAVORABLE" in response_text.upper():
                result["market_context"]["trading_conditions"] = "FAVORABLE"
            elif "NEUTRAL" in response_text.upper():
                result["market_context"]["trading_conditions"] = "NEUTRAL"
                
            return result
            
        except Exception as e:
            logging.error(f"Erreur lors du parsing de la réponse GPT: {e}")
            return {
                "decision": "HOLD",
                "confidence": 0,
                "market_context": {
                    "trend_strength": 0,
                    "volatility_rating": "LOW",
                    "trading_conditions": "UNFAVORABLE"
                }
            }

    def convert_to_json_serializable(self, obj):
        """Convertit les objets en format JSON sérialisable"""
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            elif isinstance(obj, pd.Interval):
                return str(obj)
            elif isinstance(obj, dict):
                return {str(self.convert_to_json_serializable(key)): self.convert_to_json_serializable(value) 
                        for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self.convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, pd.Series):
                return self.convert_to_json_serializable(obj.to_dict())
            elif isinstance(obj, pd.DataFrame):
                return self.convert_to_json_serializable(obj.to_dict(orient='records'))
            elif pd.isna(obj):
                return None
            return str(obj) if hasattr(obj, '__str__') else obj
            
        except Exception as e:
            logging.error(f"Erreur lors de la conversion en JSON: {e}")
            return str(obj)

    def calculate_market_score(self, candle):
        """Calcule un score de marché avancé basé sur l'analyse technique multi-indicateurs"""
        try:
            score = 0
            
            # 1. Analyse RSI (0-25)
            rsi_score = 0
            rsi = candle['RSI']
            rsi_5 = candle['RSI_5']
            rsi_21 = candle['RSI_21']
            
            # RSI principal
            if 40 <= rsi <= 60:  # Zone neutre
                rsi_score += 5
            elif (30 <= rsi < 40) or (60 < rsi <= 70):  # Zones d'alerte
                rsi_score += 10
            elif rsi < 30 or rsi > 70:  # Zones extrêmes
                rsi_score += 15
            
            # Convergence/Divergence des RSI
            if (rsi_5 > rsi and rsi > rsi_21) or (rsi_5 < rsi and rsi < rsi_21):
                rsi_score += 10  # Confirmation de tendance
            
            score += rsi_score
            
            # 2. Analyse MACD (0-25)
            macd_score = 0
            macd = candle['MACD']
            signal = candle['Signal']
            hist = candle['MACD_Hist']
            
            # Croisement ou divergence
            if abs(macd - signal) > 0:
                macd_score += 10
            
            # Force du signal
            if abs(hist) > abs(hist.mean()) * 1.5:
                macd_score += 10
            
            # Confirmation de tendance
            if (macd > 0 and signal > 0) or (macd < 0 and signal < 0):
                macd_score += 5
                
            score += macd_score
            
            # 3. Analyse Bollinger (0-25)
            bb_score = 0
            price = candle['close']
            bb_width = candle['BB_upper'] - candle['BB_lower']
            bb_width_3 = candle['BB_upper_3'] - candle['BB_lower_3']
            
            # Position par rapport aux bandes
            if price < candle['BB_lower'] or price > candle['BB_upper']:
                bb_score += 15  # Signal fort de retournement potentiel
            elif bb_width > bb_width_3:  # Expansion des bandes
                bb_score += 10
                
            # Volatilité relative
            if bb_width > candle['ATR'] * 2:
                bb_score += 10
            elif bb_width < candle['ATR']:
                bb_score += 5
                
            score += bb_score
            
            # 4. Analyse Momentum et ATR (0-25)
            momentum_score = 0
            
            # Momentum multi-périodes
            if (candle['Momentum_5'] > 0 and 
                candle['Momentum'] > 0 and 
                candle['Momentum_21'] > 0):
                momentum_score += 10  # Forte tendance haussière
            elif (candle['Momentum_5'] < 0 and 
                  candle['Momentum'] < 0 and 
                  candle['Momentum_21'] < 0):
                momentum_score += 10  # Forte tendance baissière
            
            # ATR multi-périodes
            atr_trend = (candle['ATR_5'] > candle['ATR'] and 
                        candle['ATR'] > candle['ATR_21'])
            if atr_trend:
                momentum_score += 15  # Accélération de la volatilité
                
            score += momentum_score
            
            # 5. Analyse Volume et OBV (0-25)
            volume_score = 0
            volume = candle['tick_volume']
            obv = candle['OBV']
            obv_ma = candle['OBV_MA']
            
            # Volume relatif
            avg_volume = volume.mean() if hasattr(volume, 'mean') else volume
            if volume > avg_volume * 2:
                volume_score += 10
            elif volume > avg_volume * 1.5:
                volume_score += 5
            
            # Confirmation OBV
            if obv > obv_ma:
                volume_score += 10
            
            # Tendance du volume
            volume_trend = volume > volume.shift(1) if hasattr(volume, 'shift') else True
            if volume_trend:
                volume_score += 5
                
            score += volume_score
            
            # 6. Bonus Ichimoku (0-15)
            ichimoku_score = 0
            price = candle['close']
            
            # Position par rapport au nuage
            if (price > candle['Ichimoku_Senkou_A'] and 
                price > candle['Ichimoku_Senkou_B']):
                ichimoku_score += 5  # Au-dessus du nuage
            elif (price < candle['Ichimoku_Senkou_A'] and 
                  price < candle['Ichimoku_Senkou_B']):
                ichimoku_score += 5  # En-dessous du nuage
            
            # Croisements des lignes
            if candle['Ichimoku_Tenkan'] > candle['Ichimoku_Kijun']:
                ichimoku_score += 5
            
            # Confirmation Chikou
            if candle['Ichimoku_Chikou'] > price:
                ichimoku_score += 5
                
            score += ichimoku_score
            
            # 7. Bonus ADX (0-10)
            adx_score = 0
            adx = candle['ADX']
            plus_di = candle['Plus_DI']
            minus_di = candle['Minus_DI']
            
            if adx > 25:  # Tendance forte
                adx_score += 5
                if plus_di > minus_di:
                    adx_score += 5  # Tendance haussière confirmée
                elif minus_di > plus_di:
                    adx_score += 5  # Tendance baissière confirmée
                    
            score += adx_score
            
            # Normalisation du score final (0-100)
            return min(max(score, 0), 100)
            
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du score de marché: {e}")
            return 0

    def get_recent_performance(self, symbol):
        """Récupère les performances récentes"""
        try:
            symbol_trades = [t for t in self.trade_history if t['symbol'] == symbol]
            if not symbol_trades:
                return {
                    "win_rate": 0,
                    "avg_profit": 0,
                    "total_trades": 0
                }
                
            wins = sum(1 for t in symbol_trades if t['profit'] > 0)
            total = len(symbol_trades)
            
            return {
                "win_rate": (wins / total) * 100 if total > 0 else 0,
                "avg_profit": sum(t['profit'] for t in symbol_trades) / total if total > 0 else 0,
                "total_trades": total
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des performances: {e}")
            return {
                "win_rate": 0,
                "avg_profit": 0,
                "total_trades": 0
            }

    def initialize_mt5(self):
        """Initialise la connexion à MetaTrader 5"""
        try:
            if not mt5.initialize():
                logging.error("Échec de l'initialisation de MetaTrader 5")
                return False
                
            # Connexion au compte
            if not mt5.login(int(MT5_LOGIN), MT5_PASSWORD, MT5_SERVER):
                logging.error("Échec de la connexion au compte MetaTrader 5")
                mt5.shutdown()
                return False
                
            logging.info(f"Connexion réussie - Compte #{MT5_LOGIN} {MT5_SERVER}")
            version = mt5.version()
            version_str = f"{version[0]}.{version[1]} ({version[2]})" if isinstance(version, tuple) else str(version)
            logging.info(f"MetaTrader 5 connecté: version {version_str}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de MT5: {e}")
            return False

    def setup_logging(self):
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

    def setup_telegram(self):
        """Configure la connexion Telegram"""
        try:
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                self.telegram_bot = telegram.Bot(token=TELEGRAM_TOKEN)
                logging.info("Bot Telegram initialisé")
            else:
                self.telegram_bot = None
                logging.warning("Configuration Telegram manquante")
        except Exception as e:
            logging.error(f"Erreur lors de la configuration Telegram: {e}")
            self.telegram_bot = None

    def load_or_train_model(self):
        """Charge ou entraîne le modèle de trading"""
        try:
            if os.path.exists('trading_model.pkl'):
                return joblib.load('trading_model.pkl')
            else:
                model = RandomForestClassifier(n_estimators=100)
                # Entraînement basique
                joblib.dump(model, 'trading_model.pkl')
                return model
        except Exception as e:
            logging.error(f"Erreur lors du chargement/entraînement du modèle: {e}")
            return None

    def check_market_conditions(self):
        """Vérifie les conditions du marché"""
        try:
            current_time = datetime.now().time()
            # Vérifier si le marché est ouvert (à adapter selon vos horaires)
            return True  # Pour le moment, toujours retourner True
        except Exception as e:
            logging.error(f"Erreur lors de la vérification des conditions de marché: {e}")
            return False

    def get_time_to_next_session(self):
        """Calcule le temps jusqu'à la prochaine session"""
        try:
            now = datetime.now()
            next_session = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            return (next_session - now).total_seconds()
        except Exception as e:
            logging.error(f"Erreur lors du calcul du temps jusqu'à la prochaine session: {e}")
            return 600  # 10 minutes par défaut

    def validate_trade_conditions(self, analysis, symbol):
        """Valide les conditions pour un trade avec des critères plus stricts"""
        try:
            # 1. Vérifications de base
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logging.error(f"Impossible d'obtenir les informations pour {symbol}")
                return False

            # 2. Vérification du spread avec gestion dynamique
            current_spread = symbol_info.spread
            max_spread = self.calculate_max_spread(symbol, symbol_info)
            if current_spread > max_spread:
                logging.info(f"Spread trop élevé pour {symbol}: {current_spread} > {max_spread}")
                return False

            # 3. Vérification de la force de la tendance
            if analysis.get("market_context", {}).get("trend_strength", 0) < self.min_trend_strength:
                logging.info(f"Force de tendance insuffisante pour {symbol}: {analysis.get('market_context', {}).get('trend_strength', 0)} < {self.min_trend_strength}")
                return False

            # 5. Vérification du volume
            market_data = self.get_market_data_multi_timeframe(symbol)
            if market_data and "H1" in market_data:
                df = market_data["H1"]
                avg_volume = df['tick_volume'].rolling(20).mean().iloc[-1]
                current_volume = df['tick_volume'].iloc[-1]
                if current_volume < avg_volume * self.min_volume_threshold:
                    logging.info(f"Volume insuffisant pour {symbol}")
                    return False

            # 6. Vérification du ratio risque/récompense
            if "risk_management" in analysis:
                entry_price = float(analysis["entry"]["price"])
                stop_loss = float(analysis["risk_management"]["stop_loss"])
                take_profit = float(analysis["risk_management"]["take_profit"])
                
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                
                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio < self.min_risk_reward:
                        logging.info(f"Ratio risque/récompense insuffisant pour {symbol}: {rr_ratio} < {self.min_risk_reward}")
                        return False

            # 7. Vérification des conditions de marché
            if analysis.get("market_context", {}).get("trading_conditions") == "UNFAVORABLE":
                logging.info(f"Conditions de marché défavorables pour {symbol}")
                return False

            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la validation des conditions: {e}")
            return False

    def place_trade(self, decision, symbol, suggested_size=None):
        """Place un ordre de trading"""
        try:
            if decision not in ["BUY", "SELL"]:
                return False
                
            # Calculer la taille de la position
            account_info = mt5.account_info()
            if account_info is None:
                return False
                
            position_size = suggested_size or self._calculate_position_size(symbol)
            
            # Préparer l'ordre
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            point = symbol_info.point
            price = mt5.symbol_info_tick(symbol).ask if decision == "BUY" else mt5.symbol_info_tick(symbol).bid
            
            order = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": position_size,
                "type": mt5.ORDER_TYPE_BUY if decision == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": SLIPPAGE,
                "magic": 234000,
                "comment": "order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(order)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Erreur lors du placement de l'ordre: {result.comment}")
                return False
                
            logging.info(f"Ordre placé avec succès: {symbol} {decision}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors du placement de l'ordre: {e}")
            return False

    def manage_positions(self):
        """Gère les positions ouvertes"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return
                
            for position in positions:
                # Vérifier les conditions de sortie
                if self._check_exit_conditions(position):
                    self._close_position(position)
                    
        except Exception as e:
            logging.error(f"Erreur lors de la gestion des positions: {e}")

    def _calculate_position_size(self, symbol):
        """Calcule la taille de la position avec gestion de risque adaptative"""
        try:
            account_info = mt5.account_info()
            symbol_info = mt5.symbol_info(symbol)
            
            if account_info is None or symbol_info is None:
                return 0.01  # Taille minimale par défaut
            
            # Analyse de la performance récente
            recent_trades = self.trade_history[-10:]
            consecutive_losses = 0
            for trade in reversed(recent_trades):
                if trade['profit'] < 0:
                    consecutive_losses += 1
                else:
                    break
            
            # Ajustement du risque basé sur les pertes consécutives
            risk_multiplier = max(0.5, 1 - (consecutive_losses * 0.2))  # Réduction progressive du risque
            
            # Calcul du risque adaptatif
            base_risk = RISK_PER_TRADE_PERCENT / 100
            volatility_factor = self.calculate_volatility_factor(symbol)
            market_condition_factor = self.assess_market_conditions(symbol)
            
            adjusted_risk = (base_risk * risk_multiplier * 
                           volatility_factor * market_condition_factor)
            
            # Calculer le montant risqué en fonction de l'equity
            risk_amount = account_info.equity * adjusted_risk
            
            # Obtenir le prix actuel et la valeur d'un pip
            current_price = symbol_info.ask
            pip_value = symbol_info.point * symbol_info.trade_contract_size
            
            # Calculer la valeur d'un pip en devise du compte avec gestion des cross rates améliorée
            if symbol.endswith('USD'):
                pip_value_account = pip_value
            elif symbol.startswith('USD'):
                pip_value_account = pip_value / current_price
            else:
                try:
                    # Gestion améliorée des cross rates
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:6]
                    
                    # Obtenir les taux de conversion nécessaires
                    if base_currency != 'USD':
                        base_usd = mt5.symbol_info(f"{base_currency}USD")
                        base_rate = base_usd.bid if base_usd else 1
                    else:
                        base_rate = 1
                        
                    if quote_currency != 'USD':
                        quote_usd = mt5.symbol_info(f"{quote_currency}USD")
                        quote_rate = quote_usd.bid if quote_usd else 1
                    else:
                        quote_rate = 1
                    
                    pip_value_account = pip_value * (base_rate / quote_rate)
                except:
                    # Fallback au calcul simple si erreur
                    usd_rate = mt5.symbol_info('EURUSD').bid if symbol.startswith('EUR') else mt5.symbol_info('USD' + symbol[:3]).bid
                    pip_value_account = pip_value * usd_rate
            
            # Calculer la taille de position avec stop loss dynamique
            atr = self.calculate_atr(symbol)
            stop_loss_pips = max(STOP_LOSS_PIPS, int(atr * 1.5))  # Stop loss dynamique basé sur l'ATR
            position_size = (risk_amount / (stop_loss_pips * pip_value_account))
            
            # Limites de position avec vérification de marge
            min_lot = symbol_info.volume_min
            max_lot = min(symbol_info.volume_max, 0.5)  # Limite maximale conservative
            position_size = max(min(round(position_size / min_lot) * min_lot, max_lot), min_lot)
            
            # Vérifications de sécurité supplémentaires
            margin_required = position_size * symbol_info.margin_initial
            free_margin = account_info.margin_free
            
            if margin_required > free_margin * 0.8:  # Marge de sécurité de 20%
                position_size = (free_margin * 0.8) / symbol_info.margin_initial
                position_size = max(min(round(position_size / min_lot) * min_lot, max_lot), min_lot)
            
            # Limites pour petits comptes avec scaling progressif
            if account_info.equity < 1000:
                position_size = min(position_size, 0.1)
            elif account_info.equity < 5000:
                position_size = min(position_size, 0.3)
            
            return position_size
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.01

    def _check_exit_conditions(self, position):
        """Vérifie les conditions de sortie pour une position"""
        try:
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return False
                
            current_price = symbol_info.bid if position.type == mt5.POSITION_TYPE_BUY else symbol_info.ask
            
            # Vérifier le stop loss et take profit
            if position.tp > 0 and current_price >= position.tp:
                return True
            if position.sl > 0 and current_price <= position.sl:
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Erreur lors de la vérification des conditions de sortie: {e}")
            return False

    def _close_position(self, position):
        """Ferme une position"""
        try:
            # Préparer l'ordre de fermeture
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                return False
                
            close_price = symbol_info.bid if position.type == mt5.POSITION_TYPE_BUY else symbol_info.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": position.ticket,
                "price": close_price,
                "deviation": SLIPPAGE,
                "magic": 234000,
                "comment": "close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Erreur lors de la fermeture de la position: {result.comment}")
                return False
                
            logging.info(f"Position fermée avec succès: {position.symbol}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la fermeture de la position: {e}")
            return False

    def calculate_volatility_factor(self, symbol):
        """Calcule le facteur de volatilité pour l'ajustement du risque"""
        try:
            # Récupérer les données récentes
            data = self.get_market_data_multi_timeframe(symbol)
            if not data or "H1" not in data:
                return 1.0

            df = data["H1"]
            
            # Calculer la volatilité relative
            atr = df['ATR'].iloc[-1]
            avg_price = df['close'].mean()
            volatility = (atr / avg_price) * 100
            
            # Ajuster le facteur en fonction de la volatilité
            if volatility > 1.5:  # Haute volatilité
                return 0.7  # Réduire le risque
            elif volatility < 0.5:  # Basse volatilité
                return 1.2  # Augmenter légèrement le risque
            else:
                return 1.0  # Volatilité normale
                
        except Exception as e:
            logging.error(f"Erreur lors du calcul du facteur de volatilité: {e}")
            return 1.0

    def assess_market_conditions(self, symbol):
        """Évalue les conditions de marché pour l'ajustement du risque"""
        try:
            # Récupérer les données multi-timeframes
            data = self.get_market_data_multi_timeframe(symbol)
            if not data:
                return 1.0
            
            # Analyser les différentes timeframes
            conditions_score = 0
            timeframes_analyzed = 0
            
            for tf in ["H1", "H4", "D1"]:
                if tf not in data:
                    continue
                    
                df = data[tf]
                if df is None or df.empty:
                    continue
                
                # Vérifier la tendance
                trend = self.detect_trend(df)
                if trend in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
                    conditions_score += 1
                elif trend in ["UPTREND", "DOWNTREND"]:
                    conditions_score += 0.5
                
                # Vérifier la volatilité
                volatility = self.calculate_volatility(df)
                if volatility < 30:  # Faible volatilité
                    conditions_score += 1
                elif volatility > 70:  # Haute volatilité
                    conditions_score -= 0.5
                
                # Vérifier le volume
                recent_volume = df['tick_volume'].tail(20)
                avg_volume = recent_volume.mean()
                if recent_volume.iloc[-1] > avg_volume * 1.5:
                    conditions_score += 0.5
                
                timeframes_analyzed += 1
            
            if timeframes_analyzed == 0:
                return 1.0
            
            # Calculer le score final
            final_score = conditions_score / timeframes_analyzed
            
            # Convertir le score en facteur de risque
            if final_score > 1.5:
                return 1.2  # Conditions très favorables
            elif final_score < 0:
                return 0.7  # Conditions défavorables
            else:
                return 1.0  # Conditions normales
                
        except Exception as e:
            logging.error(f"Erreur lors de l'évaluation des conditions de marché: {e}")
            return 1.0

    def calculate_max_spread(self, symbol, symbol_info):
        """Calcule le spread maximum autorisé pour un symbole donné"""
        try:
            # Récupérer les paramètres de spread pour le symbole
            spread_params = self.spread_settings.get(symbol, self.spread_settings['default'])
            
            # Calculer le facteur de volatilité
            market_data = self.get_market_data_multi_timeframe(symbol)
            volatility_multiplier = 1.0
            
            if market_data and "H1" in market_data:
                df = market_data["H1"]
                volatility = self.calculate_volatility(df)
                
                # Ajuster le multiplicateur en fonction de la volatilité
                if volatility > 70:  # Haute volatilité
                    volatility_multiplier = 1 + spread_params['volatility_factor']
                elif volatility < 30:  # Basse volatilité
                    volatility_multiplier = 1 - (spread_params['volatility_factor'] / 2)
            
            # Récupérer le spread moyen du marché
            avg_spread = MARKET_FILTERS["SPREAD_FILTER"]["max_spread_pips"]
            
            # Calculer le spread maximum autorisé
            max_spread = avg_spread * spread_params['base'] * volatility_multiplier
            
            # Ajustement pour les sessions de marché
            current_hour = datetime.now().hour
            if 22 <= current_hour or current_hour < 2:  # Session asiatique début
                max_spread *= 1.2  # Spread plus large autorisé
            elif 8 <= current_hour < 10:  # Ouverture session européenne
                max_spread *= 1.1  # Spread légèrement plus large autorisé
            
            return max_spread
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du spread maximum: {e}")
            return float('inf')  # En cas d'erreur, retourner une valeur infinie pour éviter le trade

    def calculate_atr(self, symbol):
        """Calcule l'ATR (Average True Range) pour un symbole"""
        try:
            data = self.get_market_data_multi_timeframe(symbol)
            if not data or "H1" not in data:
                return STOP_LOSS_PIPS
                
            df = data["H1"]
            return int(df['ATR'].iloc[-1] / df['point'].iloc[0])
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul de l'ATR: {e}")
            return STOP_LOSS_PIPS

    def start(self):
        """Point d'entrée principal du bot"""
        try:
            self.run()
        except KeyboardInterrupt:
            logging.info("Arrêt du bot par l'utilisateur")
        finally:
            mt5.shutdown()
            logging.info("MetaTrader 5 déconnecté")

if __name__ == "__main__":
    try:
        bot = GPTTradingBot()
        bot.start()
    except Exception as e:
        logging.error(f"Erreur fatale lors du démarrage du bot: {e}")
        mt5.shutdown()
