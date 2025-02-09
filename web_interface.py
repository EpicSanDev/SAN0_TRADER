from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
from flask_socketio import SocketIO, emit
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from config import *
import json
import logging
from trading_bot import GPTTradingBot

# Initialize Flask app and global variables
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store chat messages and initialize bot
chat_messages = []
trading_bot = GPTTradingBot()

# Initialize MT5 connection
if not mt5.initialize():
    print("Erreur d'initialisation MT5")
    mt5.shutdown()
else:
    print("MT5 initialisé avec succès")

def get_market_data_multi_timeframe(symbol):
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
            rates = mt5.copy_rates_from(symbol, tf_value, datetime.now(), 1000)
            if rates is None or len(rates) == 0:
                continue
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            result[tf_name] = df
            
        return result
            
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données multi-timeframes pour {symbol}: {e}")
        return None

def calculate_volatility(df):
    """Calcule la volatilité du marché"""
    try:
        # Utiliser l'ATR comme mesure de volatilité
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        avg_price = df['close'].mean()
        volatility = (atr / avg_price) * 100
        
        # Normaliser entre 0 et 100
        return min(max(volatility * 10, 0), 100)
        
    except Exception as e:
        logging.error(f"Erreur lors du calcul de la volatilité: {e}")
        return 0

def detect_trend(df):
    """Détecte la tendance du marché"""
    try:
        # Calculer les moyennes mobiles
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()
        ma200 = df['close'].rolling(window=200).mean()
        
        # Vérifier l'alignement des moyennes mobiles
        if ma20.iloc[-1] > ma50.iloc[-1] > ma200.iloc[-1]:
            if df['close'].iloc[-1] > ma20.iloc[-1]:
                return "STRONG_UPTREND"
            return "UPTREND"
        elif ma20.iloc[-1] < ma50.iloc[-1] < ma200.iloc[-1]:
            if df['close'].iloc[-1] < ma20.iloc[-1]:
                return "STRONG_DOWNTREND"
            return "DOWNTREND"
            
        # Vérifier les conditions de range
        bb_width = df['close'].rolling(20).std().iloc[-1] * 2
        avg_bb_width = df['close'].rolling(20).std().mean() * 2
        
        if bb_width < avg_bb_width * 0.8:
            return "RANGING_TIGHT"
        elif bb_width > avg_bb_width * 1.2:
            return "RANGING_WIDE"
        
        return "RANGING_NORMAL"
        
    except Exception as e:
        logging.error(f"Erreur lors de la détection de tendance: {e}")
        return "NEUTRAL"

def get_account_info():
    """Récupère les informations du compte"""
    try:
        account_info = mt5.account_info()
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level,
            'profit': account_info.profit
        }
    except:
        return None

def get_open_positions():
    """Récupère les positions ouvertes"""
    try:
        positions = mt5.positions_get()
        if positions:
            positions_list = []
            for pos in positions:
                positions_list.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'open_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': datetime.fromtimestamp(pos.time).strftime('%Y-%m-%d %H:%M:%S')
                })
            return positions_list
        return []
    except:
        return []

def create_price_chart(symbol, timeframe="H1", periods=100):
    """Crée un graphique des prix avec indicateurs"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, f"TIMEFRAME_{timeframe}"), 0, periods)
        if rates is None:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calcul des indicateurs
        df['SMA20'] = df['close'].rolling(window=20).mean()
        df['SMA50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['close'], 14)
        
        # Création du graphique principal
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])

        # Graphique des chandeliers
        fig.add_trace(go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)

        # Ajout des moyennes mobiles
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['SMA20'],
            name='SMA20',
            line=dict(color='orange')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['SMA50'],
            name='SMA50',
            line=dict(color='blue')
        ), row=1, col=1)

        # Ajout du RSI
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['RSI'],
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)

        # Ajout des lignes horizontales pour le RSI
        fig.add_hline(y=70, line_color="red", line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dash", row=2, col=1)

        # Mise en page
        fig.update_layout(
            title=f'{symbol} - {timeframe}',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig.to_json()

    except Exception as e:
        print(f"Erreur lors de la création du graphique: {e}")
        return None

def calculate_rsi(series, periods=14):
    """Calcule le RSI"""
    delta = series.diff()
    gain = (delta > 0) * delta
    loss = (delta < 0) * -delta
    
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_trading_stats():
    """Récupère les statistiques de trading"""
    try:
        today = datetime.now().date()
        start_date = today - timedelta(days=30)
        
        # Récupérer l'historique des trades
        history = mt5.history_deals_get(start_date, datetime.now())
        
        if history:
            trades = []
            current_trade = None
            
            for deal in history:
                if deal.entry == 0:  # Position ouverte
                    current_trade = {
                        'ticket': deal.ticket,
                        'symbol': deal.symbol,
                        'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                        'volume': deal.volume,
                        'open_price': deal.price,
                        'open_time': datetime.fromtimestamp(deal.time),
                        'commission': deal.commission,
                        'swap': 0,
                        'profit': 0
                    }
                elif deal.entry == 1 and current_trade:  # Position fermée
                    current_trade['close_price'] = deal.price
                    current_trade['close_time'] = datetime.fromtimestamp(deal.time)
                    current_trade['swap'] = deal.swap
                    current_trade['profit'] = deal.profit
                    trades.append(current_trade)
                    current_trade = None

            # Calcul des statistiques
            if trades:
                df = pd.DataFrame(trades)
                winning_trades = df[df['profit'] > 0]
                losing_trades = df[df['profit'] <= 0]

                stats = {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
                    'total_profit': df['profit'].sum(),
                    'largest_win': df['profit'].max(),
                    'largest_loss': df['profit'].min(),
                    'average_win': winning_trades['profit'].mean() if not winning_trades.empty else 0,
                    'average_loss': losing_trades['profit'].mean() if not losing_trades.empty else 0,
                    'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if not losing_trades.empty and losing_trades['profit'].sum() != 0 else 0
                }
                return stats

        return None
    except:
        return None

def place_order(symbol, order_type, volume, sl=None, tp=None):
    """Place un ordre sur le marché"""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False, "Symbole non trouvé"

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                return False, "Symbole non sélectionnable"

        point = symbol_info.point
        price = mt5.symbol_info_tick(symbol).ask if order_type == "BUY" else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 100,
            "comment": "ordre depuis interface web",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl:
            request["sl"] = float(sl)
        if tp:
            request["tp"] = float(tp)

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Erreur d'exécution: {result.comment}"
        
        return True, "Ordre exécuté avec succès"
    except Exception as e:
        return False, str(e)

def get_trade_history(days=30):
    """Récupère l'historique des trades"""
    try:
        from_date = datetime.now() - timedelta(days=days)
        history = mt5.history_deals_get(from_date, datetime.now())
        
        if history:
            trades = []
            for deal in history:
                if deal.entry == 1:  # Position fermée
                    trades.append({
                        'ticket': deal.ticket,
                        'symbol': deal.symbol,
                        'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                        'volume': deal.volume,
                        'open_price': deal.price,
                        'close_price': deal.price,
                        'close_time': datetime.fromtimestamp(deal.time).strftime('%Y-%m-%d %H:%M:%S'),
                        'profit': deal.profit
                    })
            return trades
        return []
    except:
        return []

@app.route('/')
def index():
    """Page d'accueil du tableau de bord"""
    return render_template('index.html')

@app.route('/settings')
def settings():
    """Page de configuration du bot"""
    return render_template('settings.html')

def save_config(config_data):
    """Sauvegarde la configuration dans config.py"""
    config_path = 'config.py'
    with open(config_path, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # Mettre à jour chaque paramètre
    for key, value in config_data.items():
        # Convertir les valeurs en format Python
        if isinstance(value, bool):
            value_str = str(value)
        elif isinstance(value, str):
            value_str = f'"{value}"'
        else:
            value_str = str(value)
        
        # Remplacer la valeur dans le fichier
        import re
        pattern = fr'^{key}\s*=.*$'
        replacement = f'{key} = {value_str}'
        config_content = re.sub(pattern, replacement, config_content, flags=re.MULTILINE)
    
    # Sauvegarder le fichier
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Récupère les paramètres actuels"""
    try:
        # Lire les paramètres depuis config.py
        settings = {
            'LOT_SIZE': LOT_SIZE,
            'STOP_LOSS_PIPS': STOP_LOSS_PIPS,
            'TAKE_PROFIT_PIPS': TAKE_PROFIT_PIPS,
            'TIMEFRAME': TIMEFRAME,
            'SLIPPAGE': SLIPPAGE,
            'MAX_RISK_PERCENT': MAX_RISK_PERCENT,
            'MAX_DAILY_LOSS': MAX_DAILY_LOSS,
            'MAX_TRADES_PER_DAY': MAX_TRADES_PER_DAY,
            'MAX_DRAWDOWN_PERCENT': RISK_MANAGEMENT['MAX_DRAWDOWN_PERCENT'],
            'DAILY_TARGET_PERCENT': RISK_MANAGEMENT['DAILY_TARGET_PERCENT'],
            'WEEKLY_TARGET_PERCENT': RISK_MANAGEMENT['WEEKLY_TARGET_PERCENT'],
            'RSI_PERIOD': RSI_PERIOD,
            'MACD_FAST': MACD_FAST,
            'MACD_SLOW': MACD_SLOW,
            'MACD_SIGNAL': MACD_SIGNAL,
            'MA_PERIOD': MA_PERIOD,
            'max_spread_pips': MARKET_FILTERS['SPREAD_FILTER']['max_spread_pips'],
            'min_volume_threshold': MARKET_FILTERS['VOLUME_FILTER']['min_volume_threshold'],
            'min_trend_strength': MARKET_FILTERS['TREND_FILTER']['min_trend_strength'],
            'min_correlation': MARKET_FILTERS['CORRELATION_FILTER']['min_correlation'],
            'VOLATILITY_FILTER': MARKET_FILTERS['VOLATILITY_FILTER'],
            'NEWS_FILTER': MARKET_FILTERS['NEWS_FILTER'],
            'MIN_SCORE_TO_TRADE': MIN_SCORE_TO_TRADE,
            'VOLATILITY_THRESHOLD': VOLATILITY_THRESHOLD,
            'USE_AUTO_TPSL': USE_AUTO_TPSL,
            'USE_TRAILING_STOP': USE_TRAILING_STOP
        }
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Met à jour les paramètres"""
    try:
        data = request.get_json()
        
        # Convertir les valeurs numériques
        for key in data:
            if key not in ['TIMEFRAME', 'VOLATILITY_FILTER', 'NEWS_FILTER', 'USE_AUTO_TPSL', 'USE_TRAILING_STOP']:
                try:
                    data[key] = float(data[key])
                except ValueError:
                    pass
        
        save_config(data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    """Réinitialise les paramètres aux valeurs par défaut"""
    try:
        # Copier config.template.py vers config.py
        with open('config.template.py', 'r', encoding='utf-8') as source:
            with open('config.py', 'w', encoding='utf-8') as dest:
                dest.write(source.read())
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account')
def account_info():
    """API pour les informations du compte"""
    return jsonify(get_account_info())

@app.route('/api/positions')
def positions():
    """API pour les positions ouvertes"""
    return jsonify(get_open_positions())

@app.route('/api/chart/<symbol>')
def chart(symbol):
    """API pour les graphiques"""
    return jsonify({"chart": create_price_chart(symbol)})

@app.route('/api/stats')
def stats():
    """API pour les statistiques de trading"""
    return jsonify(get_trading_stats())

@app.route('/api/order', methods=['POST'])
def order():
    """API pour placer un ordre"""
    try:
        data = request.get_json()
        success, message = place_order(
            data['symbol'],
            data['type'],
            data['volume'],
            data.get('sl'),
            data.get('tp')
        )
        return jsonify({"success": success, "message": message})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/spreads')
def spreads():
    """API pour les spreads en temps réel"""
    try:
        spreads_data = []
        for symbol in SYMBOLS:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                # Calculer le spread maximum autorisé en utilisant la logique du bot
                market_data = get_market_data_multi_timeframe(symbol)
                volatility_multiplier = 1.0
                
                if market_data and "H1" in market_data:
                    df = market_data["H1"]
                    volatility = calculate_volatility(df)
                    spread_params = {
                        'EURUSD': {'base': 1.2, 'volatility_factor': 0.2},
                        'GBPUSD': {'base': 1.3, 'volatility_factor': 0.25},
                        'USDJPY': {'base': 1.2, 'volatility_factor': 0.2},
                        'EURGBP': {'base': 1.4, 'volatility_factor': 0.3},
                        'AUDUSD': {'base': 1.5, 'volatility_factor': 0.35},
                        'USDCAD': {'base': 1.4, 'volatility_factor': 0.3},
                        'BTCUSD': {'base': 2.0, 'volatility_factor': 0.5},
                        'ETHUSD': {'base': 2.0, 'volatility_factor': 0.5},
                        'default': {'base': 1.5, 'volatility_factor': 0.4}
                    }
                    params = spread_params.get(symbol, spread_params['default'])
                    
                    if volatility > 70:
                        volatility_multiplier = 1 + params['volatility_factor']
                    elif volatility < 30:
                        volatility_multiplier = 1 - (params['volatility_factor'] / 2)
                
                avg_spread = MARKET_FILTERS["SPREAD_FILTER"]["max_spread_pips"]
                max_spread = avg_spread * params['base'] * volatility_multiplier
                
                # Ajustement pour les sessions de marché
                current_hour = datetime.now().hour
                if 22 <= current_hour or current_hour < 2:
                    max_spread *= 1.2
                elif 8 <= current_hour < 10:
                    max_spread *= 1.1
                
                spreads_data.append({
                    'symbol': symbol,
                    'current_spread': symbol_info.spread,
                    'max_spread': max_spread
                })
        return jsonify(spreads_data)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des spreads: {e}")
        return jsonify([])

@app.route('/api/market_conditions')
def market_conditions():
    """API pour les conditions de marché"""
    try:
        conditions_data = []
        for symbol in SYMBOLS:
            market_data = get_market_data_multi_timeframe(symbol)
            if market_data and "H1" in market_data:
                df = market_data["H1"]
                volatility = calculate_volatility(df)
                trend = detect_trend(df)
                volume_trend = "INCREASING" if df['tick_volume'].diff().mean() > 0 else "DECREASING"
                
                conditions_data.append({
                    'symbol': symbol,
                    'volatility': volatility,
                    'trend': trend,
                    'volume_trend': volume_trend
                })
        return jsonify(conditions_data)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des conditions de marché: {e}")
        return jsonify([])

@app.route('/api/performance')
def performance():
    """API pour les données de performance"""
    try:
        # Récupérer l'historique des trades
        history = get_trade_history(30)  # 30 derniers jours
        
        # Calculer les rendements journaliers
        daily_returns = []
        dates = []
        current_date = None
        daily_profit = 0
        
        for trade in sorted(history, key=lambda x: x['close_time']):
            trade_date = datetime.strptime(trade['close_time'], '%Y-%m-%d %H:%M:%S').date()
            if current_date is None:
                current_date = trade_date
            
            if trade_date != current_date:
                daily_returns.append(daily_profit)
                dates.append(current_date.strftime('%Y-%m-%d'))
                daily_profit = 0
                current_date = trade_date
            
            daily_profit += trade['profit']
        
        # Ajouter le dernier jour
        if current_date:
            daily_returns.append(daily_profit)
            dates.append(current_date.strftime('%Y-%m-%d'))
        
        # Calculer la distribution des profits/pertes
        profits = [trade['profit'] for trade in history]
        min_profit = min(profits) if profits else 0
        max_profit = max(profits) if profits else 0
        
        # Créer des intervalles de profit
        num_bins = 10
        profit_ranges = np.linspace(min_profit, max_profit, num_bins)
        trade_counts = np.zeros(num_bins - 1)
        
        for profit in profits:
            for i in range(len(profit_ranges) - 1):
                if profit_ranges[i] <= profit < profit_ranges[i + 1]:
                    trade_counts[i] += 1
                    break
        
        return jsonify({
            'dates': dates,
            'daily_returns': daily_returns,
            'profit_ranges': profit_ranges[:-1].tolist(),
            'trade_counts': trade_counts.tolist()
        })
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des données de performance: {e}")
        return jsonify({
            'dates': [],
            'daily_returns': [],
            'profit_ranges': [],
            'trade_counts': []
        })

@app.route('/api/alerts')
def alerts():
    """API pour les alertes et notifications"""
    try:
        alerts_data = []
        # Vérifier les spreads élevés
        for symbol in SYMBOLS:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                current_spread = symbol_info.spread
                if current_spread > MARKET_FILTERS["SPREAD_FILTER"]["max_spread_pips"] * 2:
                    alerts_data.append({
                        'type': 'warning',
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': f'Spread élevé sur {symbol}: {current_spread} pips',
                        'read': False
                    })
        
        # Vérifier les positions avec pertes importantes
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if pos.profit < -50:  # Alerte si perte > 50
                    alerts_data.append({
                        'type': 'danger',
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'message': f'Position {pos.ticket} sur {pos.symbol} en perte importante: {pos.profit:.2f}',
                        'read': False
                    })
        
        # Vérifier la marge disponible
        account_info = mt5.account_info()
        if account_info:
            margin_level = account_info.margin_level if account_info.margin_level else 0
            if margin_level < 150:  # Alerte si niveau de marge < 150%
                alerts_data.append({
                    'type': 'danger',
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'message': f'Niveau de marge bas: {margin_level:.2f}%',
                    'read': False
                })
        
        return jsonify(alerts_data)
    except Exception as e:
        logging.error(f"Erreur lors de la récupération des alertes: {e}")
        return jsonify([])

@app.route('/api/history')
def history():
    """API pour l'historique des trades"""
    return jsonify(get_trade_history())

# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Client WebSocket connection handler"""
    emit('chat_history', chat_messages)

@socketio.on('chat_message')
def handle_message(data):
    """Handle incoming chat messages"""
    try:
        user_message = {
            'user': data.get('user', 'Anonymous'),
            'message': data.get('message', ''),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
        chat_messages.append(user_message)
        emit('chat_message', user_message, broadcast=True)
        
        # Check MT5 connection
        if not mt5.initialize():
            error_message = {
                'user': 'System',
                'message': "Connexion à MetaTrader 5 impossible. Veuillez vérifier que MT5 est en cours d'exécution.",
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            chat_messages.append(error_message)
            emit('chat_message', error_message, broadcast=True)
            return
        
        # Generate bot response
        try:
            # Select EURUSD symbol
            if not mt5.symbol_select("EURUSD", True):
                error_message = {
                    'user': 'System',
                    'message': "Impossible de sélectionner EURUSD. Veuillez vérifier vos symboles disponibles.",
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                chat_messages.append(error_message)
                emit('chat_message', error_message, broadcast=True)
                return
            
            # Get market data
            market_data = get_market_data_multi_timeframe("EURUSD")
            if market_data and "H1" in market_data:
                df = market_data["H1"]
                trend = detect_trend(df)
                volatility = calculate_volatility(df)
                
                bot_message = {
                    'user': 'System',
                    'message': f"Analyse du marché EURUSD:\n" +
                              f"• Tendance: {trend}\n" +
                              f"• Volatilité: {volatility}%\n" +
                              f"• Prix actuel: {df['close'].iloc[-1]:.5f}",
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                
                chat_messages.append(bot_message)
                if len(chat_messages) > 100:  # Keep only last 100 messages
                    chat_messages = chat_messages[-100:]
                    
                # Emit both messages
                emit('chat_message', user_message, broadcast=True)
                emit('chat_message', bot_message, broadcast=True)
            else:
                error_message = {
                    'user': 'System',
                    'message': "Désolé, je ne peux pas obtenir les données du marché pour le moment.",
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
                chat_messages.append(error_message)
                emit('chat_message', user_message, broadcast=True)
                emit('chat_message', error_message, broadcast=True)
                
        except Exception as e:
            error_message = {
                'user': 'System',
                'message': "Désolé, une erreur s'est produite lors de l'analyse.",
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
            chat_messages.append(error_message)
            emit('chat_message', user_message, broadcast=True)
            emit('chat_message', error_message, broadcast=True)
            logging.error(f"Erreur lors de l'analyse: {e}")
            
    except Exception as e:
        logging.error(f"Erreur lors du traitement du message: {e}")

@socketio.on('request_analysis')
def handle_analysis_request(data):
    """Handle trading analysis requests"""
    try:
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # Get market data
        market_data = get_market_data_multi_timeframe(symbol)
        if not market_data or "H1" not in market_data:
            emit('analysis_response', {'error': f'Données non disponibles pour {symbol}'})
            return
            
        # Create trading bot instance for analysis
        analysis = trading_bot.analyze_with_ollama(market_data, symbol)
        
        # Send analysis back to client
        emit('analysis_response', {
            'symbol': symbol,
            'analysis': analysis,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        logging.error(f"Erreur lors de l'analyse: {e}")
        emit('analysis_response', {'error': str(e)})

if __name__ == '__main__':
    if not mt5.initialize():
        print("Erreur d'initialisation MT5")
    else:
        socketio.run(app, debug=True, port=5000)
