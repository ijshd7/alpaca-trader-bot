import os
import time
import logging
import gc
import sys
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

gc.enable()

load_dotenv()

class TraderBot:
    def __init__(self):
        self.PAPER_API_BASE_URL = os.getenv('PAPER_API_BASE_URL')
        self.PAPER_API_KEY = os.getenv("PAPER_API_KEY")
        self.PAPER_API_SECRET = os.getenv("PAPER_API_SECRET")

        self.CASH_ALLOCATION = 0.15
        self.STOP_LOSS_PERCENTAGE = 0.02
        self.TAKE_PROFIT_PERCENTAGE = 0.04
        self.MAX_POSITION_SIZE = 0.5
        self.ATR_STOP_LOSS_MULTIPLIER = 1.5
        self.ATR_TAKE_PROFIT_MULTIPLIER = 3.5
        self.MIN_TRADE_COOLDOWN = 3600
        self.MAX_TRADE_COOLDOWN = 86400
        self.MAX_PORTFOLIO_EXPOSURE = 0.95
        self.REST_INTERVAL = 3600
        self.SCAN_INTERVAL = 900
        self.TRADE_COOLDOWN = 28800
        self.SYMBOLS_TO_TRADE = [
            "NVDA", "TSLA", "AMD", "META", "AAPL",
            "MSFT", "GOOGL", "NFLX", "AMZN", "INTC",
            "SPY", "QQQ", "PG", "JNJ"
        ]
        self.SYMBOLS_TO_BACKTEST = [
            "NVDA", "TSLA", "AMD", "META", "AAPL",
            "MSFT", "GOOGL", "NFLX", "AMZN", "INTC",
            "SPY", "QQQ", "PG", "JNJ"
        ]

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

        self.trading_client = TradingClient(self.PAPER_API_KEY, self.PAPER_API_SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(self.PAPER_API_KEY, self.PAPER_API_SECRET)

    def is_market_open(self):
        return self.trading_client.get_clock().is_open

    def get_market_clock(self):
        return self.trading_client.get_clock()

    def get_position(self, symbol):
        positions = self.trading_client.get_all_positions()
        for position in positions:
            if position.symbol == symbol:
                return float(position.qty)
        return 0
    
    def get_order_request_by_symbol(self, symbol):
        request_params = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            side=OrderSide.BUY,
            symbols=[symbol],
            limit=1
        )
        order = self.trading_client.get_orders(filter=request_params)
        return order

    def get_stock_data(self, symbol, limit=5000):
        now = datetime.now()
        two_years_ago = now - relativedelta(years=2)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=two_years_ago,
            limit=limit
        )
        df = self.data_client.get_stock_bars(request).df
        return self.compute_technical_indicators(df)
    
    def get_stock_data_for_backtest(self, symbol, limit=20000, years_back=4):
        now = datetime.now()
        years_ago = now - relativedelta(years=years_back)
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(1, TimeFrameUnit.Hour),
            start=years_ago,
            limit=limit
        )
        df = self.data_client.get_stock_bars(request).df
        return self.compute_technical_indicators(df)
    
    def get_latest_price(self, symbol):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                limit=1
            )
            bars = self.data_client.get_stock_bars(request).df
            if not bars.empty:
                latest_price = bars["close"].iloc[-1]
                return float(latest_price)
            else:
                logging.warning(f"⚠️ No price data found for {symbol}.")
                return None
        except Exception as e:
            logging.error(f"❌ Failed to fetch latest price for {symbol}: {e}")
            return None

    def compute_technical_indicators(self, df, window=20):
        df["True_Range"] = (df[["high", "low", "close"]].shift()).max(axis=1) - (df[["high", "low", "close"]].shift()).min(axis=1)
        df["ATR_14"] = df["True_Range"].rolling(window=14).mean()
        df["ATR_50"] = df["True_Range"].rolling(window=50).mean()
        
        median_atr = df["ATR_50"].median()
        volatility_factor = df["ATR_50"].iloc[-1] / median_atr
        base_sma_window = window
        base_rsi_period = 14
        min_sma_window, max_sma_window = 10, 50
        min_rsi_period, max_rsi_period = 7, 21
        sma_window = base_sma_window / volatility_factor
        rsi_period = base_rsi_period / volatility_factor
        sma_window = int(max(min_sma_window, min(max_sma_window, sma_window)))
        rsi_period = int(max(min_rsi_period, min(max_rsi_period, rsi_period)))
        
        df["SMA"] = df["close"].rolling(window=sma_window).mean()
        df["StdDev"] = df["close"].rolling(window=sma_window).std()
        df["Bollinger_Upper"] = df["SMA"] + (df["StdDev"] * 2)
        df["Bollinger_Lower"] = df["SMA"] - (df["StdDev"] * 2)
        df["BB_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / df["SMA"]

        df["Momentum_5"] = df["close"].pct_change(periods=5) * 100
        df["Momentum_3"] = df["close"].pct_change(periods=3) * 100
        df["RSI"] = self.compute_rsi(df["close"], period=rsi_period)
        return df

    def compute_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def is_buy_signal(self, row, df=None):
        is_below_bollinger_lower = row["close"] < row["Bollinger_Lower"]
        is_atr_within_normal_range = row["ATR_14"] < row["ATR_50"] * 1.5
        
        if df is not None:
            rsi_threshold = df["RSI"].quantile(0.10)
        else:
            rsi_threshold = 30
        is_rsi_oversold = row["RSI"] < rsi_threshold

        momentum_threshold = -0.5
        is_momentum_not_downtrend = row["Momentum_5"] > momentum_threshold

        return (
            is_below_bollinger_lower and
            is_atr_within_normal_range and
            is_rsi_oversold and
            is_momentum_not_downtrend
        )

    def can_buy(self, symbol, trade_qty):
        last_buy_time = self.get_order_request_by_symbol(symbol)[0].filled_at

        if last_buy_time and (datetime.now(timezone.utc) - last_buy_time).seconds < self.TRADE_COOLDOWN:
            logging.info(f"⏳ Skipping {symbol} buy: recent purchase within trade cooldown period.")
            return False

        position = self.get_position(symbol)
        account = self.trading_client.get_account()
        equity = float(account.equity)
        latest_price = self.get_latest_price(symbol)
        position_value = position * latest_price if latest_price else 0

        total_exposure = sum([self.get_position(s) * self.get_latest_price(s) for s in self.SYMBOLS_TO_TRADE if self.get_latest_price(s)])
        if total_exposure is None or latest_price is None:
            logging.warning(f"🚫 Unable to calculate exposure for {symbol}. Skipping buy.")
            return False
        if (total_exposure + (latest_price * trade_qty)) / equity > self.MAX_PORTFOLIO_EXPOSURE:
            logging.warning(f"🚫 Portfolio exposure exceeds {self.MAX_PORTFOLIO_EXPOSURE * 100}%. Skipping buy.")
            return False

        if position_value / equity > self.MAX_POSITION_SIZE:
            logging.warning(f"🚫 {symbol} already exceeds {self.MAX_POSITION_SIZE * 100}% of portfolio. Skipping buy.")
            return False
        return True

    def place_order(self, symbol, side, qty):
        retries = 3
        for attempt in range(retries):
            try:
                order = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
                self.trading_client.submit_order(order_data=order)
                logging.info(f"✅ {side.upper()} Order Placed: {qty} shares of {symbol}")
                break
            except Exception as e:
                logging.error(f"❌ Attempt {attempt+1}/{retries} - {side.upper()} {qty} {symbol} Order Failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5)

    def close_positions_before_market_close(self):
        clock = self.get_market_clock()
        minutes_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60
        if minutes_to_close < 15:
            logging.info("🚨 Closing all positions before market close.")
            for position in self.trading_client.get_all_positions():
                try:
                    self.place_order(position.symbol, OrderSide.SELL, float(position.qty))
                except Exception as e:
                    logging.error(f"❌ Sell Order Failed while trying to sell {position.qty} {position.symbol}: {e}")

    def mean_reversion_trade(self, symbol):
        df = self.get_stock_data(symbol, limit=2500)
        latest = df.iloc[-1]

        position = self.get_position(symbol)
        buying_power = float(self.trading_client.get_account().buying_power)

        atr_factor = latest["ATR_14"] / latest["close"]
        cash_allocation = self.CASH_ALLOCATION * (1 / (1 + atr_factor))
        trade_qty = int((buying_power * cash_allocation) / latest["close"])
        market_is_open = self.is_market_open()

        logging.info(f"🔍🔍🔍 {symbol}...")

        if trade_qty == 0 or buying_power < latest["close"] * trade_qty:
            logging.warning(f"⚠️ Not enough buying power for {symbol}. Skipping trade.")
            return
        
        volatility_ratio = latest["ATR_14"] / latest["ATR_50"]
        base_multiplier = 1.5
        dynamic_multiplier = base_multiplier * volatility_ratio
        self.ATR_STOP_LOSS_MULTIPLIER = max(1.0, min(3.0, dynamic_multiplier))
        
        if market_is_open and self.is_buy_signal(latest, df):
            logging.info(f"📉 {symbol} BUY signal: Close={latest['close']:.2f}, Bollinger_Lower={latest['Bollinger_Lower']:.2f}, RSI={latest['RSI']:.2f}")

            if self.can_buy(symbol, trade_qty):
                if trade_qty > 0 and latest["BB_Width"] > df["BB_Width"].rolling(20).mean().iloc[-1] * 1.5:
                    logging.info(f"⚠️ Volatility is high for {symbol}. Reducing position sizing.")
                    trade_qty = int(trade_qty * 0.5)

                self.place_order(symbol, OrderSide.BUY, trade_qty)
                self.TRADE_COOLDOWN = int(self.MIN_TRADE_COOLDOWN + (self.MAX_TRADE_COOLDOWN - self.MIN_TRADE_COOLDOWN) * atr_factor)
            else:
                logging.info(f"🚫 Skipping {symbol} buy due to trade restrictions.")
        elif market_is_open and position > 0:
            avg_entry_price = float(self.trading_client.get_open_position(symbol).avg_entry_price)
            current_price = latest["close"]
            
            stop_loss_price = avg_entry_price - (latest["ATR_14"] * self.ATR_STOP_LOSS_MULTIPLIER)
            take_profit_price = avg_entry_price + (latest["ATR_14"] * self.ATR_TAKE_PROFIT_MULTIPLIER)
            
            filled_order = self.get_order_request_by_symbol(symbol)[0]
            entry_date = filled_order.filled_at.date()
            current_date = datetime.now().date()
            is_new_day = current_date > entry_date

            highest_price = max(float(filled_order.filled_avg_price), float(current_price))
            trailing_stop = highest_price - (latest["ATR_14"] * 1.5)
            
            if is_new_day:
                if current_price <= trailing_stop:
                    logging.info(f"🚨 {symbol} hit TRAILING STOP! Selling at {current_price:.2f}")
                    self.place_order(symbol, OrderSide.SELL, position)
                elif current_price <= stop_loss_price:
                    logging.info(f"🚨 {symbol} hit STOP-LOSS! Selling at {current_price:.2f}")
                    self.place_order(symbol, OrderSide.SELL, position)
                elif current_price >= take_profit_price:
                    logging.info(f"🎯 {symbol} hit TAKE-PROFIT! Selling at {current_price:.2f}")
                    self.place_order(symbol, OrderSide.SELL, position)
                elif current_price > latest["SMA"]:
                    logging.info(f"📈 {symbol} reverted to SMA - SELL signal!")
                    self.place_order(symbol, OrderSide.SELL, position)
                elif current_price > latest["Bollinger_Upper"] and latest["RSI"] > df["RSI"].quantile(0.90):
                    logging.info(f"📈 {symbol} above Bollinger Upper with overbought RSI - SELL signal!")
                    self.place_order(symbol, OrderSide.SELL, position)
                else:
                    logging.info(f"🔄 Holding {symbol}. No trade action required.")
            else:
                logging.info(f"🔒 Holding {symbol}. Same-day selling prevented (Entry: {entry_date}, Current: {current_date})")
        else:
            logging.info(f"🔄 No trade signal met for {symbol}. Holding position: {position}")

    def backtest(self, df, symbol, initial_balance=100000, years_back=4):
        balance = initial_balance
        positions = {}
        cash = balance
        trades = []
        trade_results = []
        COMMISSION_PER_SHARE = 0.01
        SLIPPAGE_PERCENT = 0.0005

        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df = df.rename(columns={"level_1": "timestamp"})
            df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.between_time('09:30', '16:00')

        for i in range(1, len(df)):
            row = df.iloc[i]
            current_date = row.name.date()

            if self.is_buy_signal(row, df):
                if cash > 0:
                    atr_factor = row["ATR_14"] / row["close"]
                    cash_allocation = self.CASH_ALLOCATION * (1 / (1 + atr_factor))
                    trade_size = int((cash * cash_allocation) / row["close"])
                    if trade_size > 0 and (trade_size * row["close"] / balance) <= self.MAX_POSITION_SIZE:
                        cost = trade_size * row["close"] * (1 + SLIPPAGE_PERCENT) + (trade_size * COMMISSION_PER_SHARE)
                        if cost <= cash:
                            positions[symbol] = {
                                "qty": trade_size,
                                "entry_price": row["close"],
                                "entry_date": current_date,
                                "highest_price": row["close"]
                            }
                            cash -= cost
                            trades.append(f"Buy {trade_size} @ {row['close']} on {current_date}")

            if symbol in positions:
                qty = positions[symbol]["qty"]
                entry_price = positions[symbol]["entry_price"]
                entry_date = positions[symbol]["entry_date"]
                is_new_day = current_date > entry_date

                volatility_ratio = row["ATR_14"] / row["ATR_50"]
                base_multiplier = 1.5
                dynamic_multiplier = base_multiplier * volatility_ratio
                self.ATR_STOP_LOSS_MULTIPLIER = max(1.0, min(3.0, dynamic_multiplier))

                stop_loss_price = entry_price - (row["ATR_14"] * self.ATR_STOP_LOSS_MULTIPLIER)
                take_profit_price = entry_price + (row["ATR_14"] * self.ATR_TAKE_PROFIT_MULTIPLIER)
                highest_price = positions[symbol]["highest_price"]
                highest_price = max(highest_price, row["close"])
                positions[symbol]["highest_price"] = highest_price
                trailing_stop = highest_price - (row["ATR_14"] * 1.5)

                if is_new_day:
                    if row["close"] <= trailing_stop:
                        proceeds = qty * row["close"] * (1 - SLIPPAGE_PERCENT) - (qty * COMMISSION_PER_SHARE)
                        balance += proceeds
                        trade_results.append((entry_price, row["close"], "Trailing Stop"))
                        del positions[symbol]
                        trades.append(f"Sell {qty} @ {row['close']} Trailing Stop on {current_date}")
                    elif row["close"] <= stop_loss_price:
                        proceeds = qty * row["close"] * (1 - SLIPPAGE_PERCENT) - (qty * COMMISSION_PER_SHARE)
                        balance += proceeds
                        trade_results.append((entry_price, row["close"], "Stop-Loss"))
                        del positions[symbol]
                        trades.append(f"Sell {qty} @ {row['close']} Stop-Loss on {current_date}")
                    elif row["close"] >= take_profit_price:
                        proceeds = qty * row["close"] * (1 - SLIPPAGE_PERCENT) - (qty * COMMISSION_PER_SHARE)
                        balance += proceeds
                        trade_results.append((entry_price, row["close"], "Take-Profit"))
                        del positions[symbol]
                        trades.append(f"Sell {qty} @ {row['close']} Take-Profit on {current_date}")
                    elif row["close"] > row["SMA"]:
                        proceeds = qty * row["close"] * (1 - SLIPPAGE_PERCENT) - (qty * COMMISSION_PER_SHARE)
                        balance += proceeds
                        trade_results.append((entry_price, row["close"], "SMA Reversion"))
                        del positions[symbol]
                        trades.append(f"Sell {qty} @ {row['close']} SMA Reversion on {current_date}")
                    elif row["close"] > row["Bollinger_Upper"]:
                        proceeds = qty * row["close"] * (1 - SLIPPAGE_PERCENT) - (qty * COMMISSION_PER_SHARE)
                        balance += proceeds
                        trade_results.append((entry_price, row["close"], "Bollinger Upper"))
                        del positions[symbol]
                        trades.append(f"Sell {qty} @ {row['close']} Bollinger Upper on {current_date}")

        for sym, pos in positions.items():
            balance += pos["qty"] * df.iloc[-1]["close"] * (1 - SLIPPAGE_PERCENT) - (pos["qty"] * COMMISSION_PER_SHARE)
            trades.append(f"Close {pos['qty']} @ {df.iloc[-1]['close']} at end of backtest")

        total_return = (balance - initial_balance) / initial_balance * 100
        win_rate = len([r for r in trade_results if r[2] in ["Take-Profit", "Bollinger Upper"]]) / len(trade_results) if trade_results else 0
        max_drawdown = self.calculate_max_drawdown(df)

        results = {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "final_balance": balance,
            "trades": trades,
            "trade_results": trade_results
        }

        logging.info(f"{years_back} Year Backtest Complete for {symbol}: Starting Balance = {initial_balance} | Final Balance = {balance:.2f} | Total Return = {total_return:.2f}%")
        logging.info(f"Total trades: {len(trades)}, Trade results: {len(trade_results)}")
        return results

    def calculate_max_drawdown(self, df):
        df['returns'] = df['close'].pct_change()
        df['cum_returns'] = (1 + df['returns']).cumprod()
        df['peak'] = df['cum_returns'].cummax()
        df['drawdown'] = (df['cum_returns'] - df['peak']) / df['peak']
        max_drawdown = df['drawdown'].min()

        return max_drawdown

    def execute_backtest(self, symbol="AAPL", initial_balance=100000, limit=20000, years_back=4):
        historical_data = self.get_stock_data_for_backtest(symbol, limit, years_back)
        self.backtest(historical_data, symbol, initial_balance, years_back)

    def execute_all_backtests(self, initial_balance=100000, limit=20000, years_back=4):
        for symbol in self.SYMBOLS_TO_BACKTEST:
            self.execute_backtest(symbol, initial_balance, limit, years_back)

    def run_mean_reversion_bot(self, backtest=False):
        while True:
            if backtest:
                self.execute_all_backtests(initial_balance=100000, limit=45000, years_back=9)
            else:
                if self.is_market_open():
                    for symbol in self.SYMBOLS_TO_TRADE:
                        self.mean_reversion_trade(symbol)
                else:
                    logging.info("❌ Market is currently closed.")
                
                if self.is_market_open():
                    time.sleep(self.SCAN_INTERVAL)
                else:
                    time.sleep(self.REST_INTERVAL)

if __name__ == "__main__":
    bot = TraderBot()
    bot.run_mean_reversion_bot(backtest=True)