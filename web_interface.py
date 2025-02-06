from flask import Flask, render_template, jsonify, request
import MetaTrader5 as mt5
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from config import *
import json

app = Flask(__name__)

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

@app.route('/api/history')
def history():
    """API pour l'historique des trades"""
    return jsonify(get_trade_history())

if __name__ == '__main__':
    if not mt5.initialize():
        print("Erreur d'initialisation MT5")
    else:
        app.run(debug=True, port=5000)
