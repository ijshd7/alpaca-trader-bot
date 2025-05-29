# 📈 Trader Bot using Alpaca Paper API

This project is a simple Python-based trading bot that interacts with the [Alpaca Paper Trading API](https://alpaca.markets/docs/api-references/trading-api/) to simulate trading strategies in a risk-free environment.

## 🚀 Features

- Connects to the Alpaca Paper Trading API
- Executes trades via `trader_bot.py`
- Easy configuration via environment variables
- Lightweight dependencies managed through `requirements.txt`

---

## 🔧 Requirements

- Python 3.7+
- An Alpaca Paper Trading Account ([signup here](https://alpaca.markets/))
- API credentials in `.env` (PAPER_API_BASE_URL, PAPER_API_KEY, and PAPER_API_SECRET)

---

## 🛠️ Setup Instructions

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python3 trader_bot.py`

## NOTE:
To enable backtesting, use `backtest` boolean arg 
