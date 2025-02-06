# AI-Powered Forex Trading Bot

An advanced automated trading system that combines technical analysis, machine learning, and real-time market sentiment analysis to execute trades on the MetaTrader 5 platform.

## Features

- **Multi-Currency Support**: Trades multiple forex pairs with customizable settings for each
- **Advanced Technical Analysis**:
  - Multiple timeframe analysis
  - Custom indicator combinations
  - Pattern recognition (candlestick patterns)
  - Support/Resistance detection
- **Risk Management**:
  - Dynamic position sizing
  - Advanced stop-loss mechanisms
  - Drawdown protection
  - Multi-level take-profit targets
- **Market Analysis**:
  - Real-time news sentiment analysis
  - Economic calendar integration
  - Volatility regime detection
  - Market condition filtering
- **Machine Learning Integration**:
  - Pattern recognition
  - Market regime classification
  - Risk optimization
- **Monitoring & Notifications**:
  - Telegram integration for trade notifications
  - Detailed logging system
  - Performance tracking

## Requirements

- Python 3.8+
- MetaTrader 5 platform installed
- Required Python packages (see requirements.txt)
- Ollama (Deepseek-R1)
## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config.template.py config.py
```
Then edit `config.py` with your settings:
- MetaTrader 5 credentials
- API keys (OpenAI, NewsAPI)
- Trading parameters
- Risk management settings

## Configuration

The bot uses a configuration file (`config.py`) for all settings. Key areas to configure:

- **Trading Parameters**: Symbols, timeframes, lot sizes
- **Risk Management**: Maximum risk per trade, daily loss limits
- **API Keys**: Required for news analysis and AI features
- **Telegram Settings**: For trade notifications (optional)

See `config.template.py` for a complete list of configurable parameters.

## Usage

1. Ensure MetaTrader 5 is running and logged in
2. Start the bot:
```bash
python trading_bot.py
```

The bot will:
- Initialize connection with MT5
- Load configurations
- Start monitoring markets
- Execute trades based on strategy conditions
- Send notifications via Telegram (if configured)

## Safety Features

- Maximum daily loss limits
- Position size restrictions
- Spread checks
- Slippage protection
- Market condition filters
- News event avoidance

## Project Structure

```
├── trading_bot.py        # Main bot implementation
├── config.py            # Configuration settings
├── news_analyzer.py     # News analysis module
├── economic_calendar.py # Economic calendar integration
├── web_interface.py    # Web interface module
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## Risk Warning

This software is for educational purposes only. Trading forex carries significant risks and may not be suitable for all investors. You can lose more than your initial investment. Please ensure you understand these risks before using this system.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This bot is provided as-is without any guarantees. The developers are not responsible for any financial losses incurred through its use. Always test thoroughly on a demo account first and never risk money you cannot afford to lose.
