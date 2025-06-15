import time
import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Configuración inicial
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Binance_MA_Bot')

# Cargar configuración
def load_config():
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        config = {
            "API_KEY": "TU_API_KEY",
            "API_SECRET": "TU_API_SECRET",
            "symbol": "BTCUSDT",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "timeframe": "1h",
            "short_ma": 18,
            "long_ma": 40,
            "risk_per_trade": 1.0,
            "take_profit": 2.0,
            "stop_loss": 1.0,
            "trailing_stop": False,
            "trailing_stop_activation": 0.5,
            "trailing_stop_distance": 0.3,
            "max_active_trades": 3,
            "trade_slippage": 0.1,
            "test_mode": True
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return config

config = load_config()

# Inicializar cliente Binance
client = Client(config['API_KEY'], config['API_SECRET'], testnet=config['test_mode'])

class TradingBot:
    def __init__(self):
        self.symbol = config['symbol']
        self.timeframe = config['timeframe']
        self.short_ma = config['short_ma']
        self.long_ma = config['long_ma']
        self.risk_per_trade = config['risk_per_trade']
        self.take_profit = config['take_profit']
        self.stop_loss = config['stop_loss']
        self.trailing_stop = config['trailing_stop']
        self.active_trades = []
        self.equity = []
        self.trade_history = []
        self.last_check = None
        self.setup_folders()

    def setup_folders(self):
        os.makedirs('logs', exist_ok=True)
        os.makedirs('charts', exist_ok=True)
        os.makedirs('data', exist_ok=True)

    def get_account_balance(self, asset):
        try:
            balance = client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except BinanceAPIException as e:
            logger.error(f"Error al obtener balance: {e}")
            return 0

    def get_current_price(self):
        try:
            ticker = client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error al obtener precio: {e}")
            return None

    def get_precision(self):
        try:
            info = client.get_symbol_info(self.symbol)
            for filt in info['filters']:
                if filt['filterType'] == 'LOT_SIZE':
                    return int(round(-np.log10(float(filt['stepSize'])), 0))
        except BinanceAPIException as e:
            logger.error(f"Error al obtener precisión: {e}")
            return 4  # Valor por defecto

    def calculate_position_size(self, price):
        balance = self.get_account_balance(config['quote_currency'])
        risk_amount = balance * (self.risk_per_trade / 100)
        stop_loss_amount = price * (self.stop_loss / 100)
        position_size = risk_amount / stop_loss_amount
        return round(position_size, self.get_precision())

    def get_historical_data(self, limit=100):
        try:
            klines = client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            data = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data['close'] = data['close'].astype(float)
            data['high'] = data['high'].astype(float)
            data['low'] = data['low'].astype(float)
            data['open'] = data['open'].astype(float)
            return data
        except BinanceAPIException as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return None

    def calculate_indicators(self, data):
        data['short_ma'] = data['close'].rolling(window=self.short_ma).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_ma).mean()
        data['rsi'] = self.calculate_rsi(data['close'], 14)
        data['atr'] = self.calculate_atr(data, 14)
        return data.dropna()

    def calculate_rsi(self, series, period):
        delta = series.diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, data, period):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()

    def generate_signal(self, data):
        last_row = data.iloc[-1]
        prev_row = data.iloc[-2]
        
        # Condiciones para compra
        buy_condition = (
            prev_row['short_ma'] <= prev_row['long_ma'] and 
            last_row['short_ma'] > last_row['long_ma'] and
            last_row['rsi'] < 70
        )
        
        # Condiciones para venta
        sell_condition = (
            prev_row['short_ma'] >= prev_row['long_ma'] and 
            last_row['short_ma'] < last_row['long_ma'] and
            last_row['rsi'] > 30
        )
        
        if buy_condition:
            return 'BUY'
        elif sell_condition:
            return 'SELL'
        else:
            return 'HOLD'

    def execute_trade(self, signal):
        if len(self.active_trades) >= config['max_active_trades']:
            logger.info("Máximo de trades activos alcanzado")
            return
            
        price = self.get_current_price()
        if price is None:
            return
            
        quantity = self.calculate_position_size(price)
        if quantity <= 0:
            logger.warning("Cantidad inválida para operar")
            return
            
        try:
            if signal == 'BUY':
                order = client.create_order(
                    symbol=self.symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                take_profit_price = round(price * (1 + self.take_profit / 100), 2)
                stop_loss_price = round(price * (1 - self.stop_loss / 100), 2)
                
                trade = {
                    'entry_price': price,
                    'quantity': quantity,
                    'take_profit': take_profit_price,
                    'stop_loss': stop_loss_price,
                    'entry_time': datetime.now(),
                    'direction': 'LONG'
                }
                self.active_trades.append(trade)
                logger.info(f"Orden de COMPRA ejecutada: {quantity} {self.symbol} a {price}")
                
            elif signal == 'SELL' and self.get_account_balance(config['base_currency']) >= quantity:
                order = client.create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                take_profit_price = round(price * (1 - self.take_profit / 100), 2)
                stop_loss_price = round(price * (1 + self.stop_loss / 100), 2)
                
                trade = {
                    'entry_price': price,
                    'quantity': quantity,
                    'take_profit': take_profit_price,
                    'stop_loss': stop_loss_price,
                    'entry_time': datetime.now(),
                    'direction': 'SHORT'
                }
                self.active_trades.append(trade)
                logger.info(f"Orden de VENTA ejecutada: {quantity} {self.symbol} a {price}")
                
            self.save_trade_data(order, signal)
            
        except BinanceAPIException as e:
            logger.error(f"Error al ejecutar orden: {e}")

    def check_take_profit_stop_loss(self):
        current_price = self.get_current_price()
        if current_price is None:
            return
            
        for trade in self.active_trades[:]:
            try:
                if trade['direction'] == 'LONG':
                    # Check Take Profit
                    if current_price >= trade['take_profit']:
                        self.close_trade(trade, 'TP', current_price)
                    # Check Stop Loss
                    elif current_price <= trade['stop_loss']:
                        self.close_trade(trade, 'SL', current_price)
                    # Trailing Stop
                    elif self.trailing_stop:
                        new_sl = current_price * (1 - self.trailing_stop_distance / 100)
                        if new_sl > trade['stop_loss']:
                            trade['stop_loss'] = new_sl
                            
                elif trade['direction'] == 'SHORT':
                    # Check Take Profit
                    if current_price <= trade['take_profit']:
                        self.close_trade(trade, 'TP', current_price)
                    # Check Stop Loss
                    elif current_price >= trade['stop_loss']:
                        self.close_trade(trade, 'SL', current_price)
                    # Trailing Stop
                    elif self.trailing_stop:
                        new_sl = current_price * (1 + self.trailing_stop_distance / 100)
                        if new_sl < trade['stop_loss']:
                            trade['stop_loss'] = new_sl
                            
            except Exception as e:
                logger.error(f"Error al verificar TP/SL: {e}")

    def close_trade(self, trade, reason, current_price):
        try:
            if trade['direction'] == 'LONG':
                side = SIDE_SELL
            else:
                side = SIDE_BUY
                
            order = client.create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=trade['quantity']
            )
            
            profit_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100
            if trade['direction'] == 'SHORT':
                profit_pct *= -1
                
            trade_info = {
                'entry_price': trade['entry_price'],
                'exit_price': current_price,
                'quantity': trade['quantity'],
                'profit_pct': profit_pct,
                'duration': (datetime.now() - trade['entry_time']).total_seconds() / 60,
                'reason': reason,
                'direction': trade['direction'],
                'exit_time': datetime.now()
            }
            
            self.trade_history.append(trade_info)
            self.active_trades.remove(trade)
            logger.info(f"Trade cerrado por {reason}. Ganancia: {profit_pct:.2f}%")
            self.save_trade_data(order, 'CLOSE', trade_info)
            
        except BinanceAPIException as e:
            logger.error(f"Error al cerrar trade: {e}")

    def save_trade_data(self, order, signal, trade_info=None):
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'signal': signal,
                'order': order,
                'trade_info': trade_info
            }
            
            log_file = f"logs/trades_{datetime.now().strftime('%Y%m%d')}.json"
            mode = 'a' if os.path.exists(log_file) else 'w'
            
            with open(log_file, mode) as f:
                if mode == 'a':
                    f.write('\n')
                json.dump(log_entry, f)
                
        except Exception as e:
            logger.error(f"Error al guardar datos del trade: {e}")

    def generate_report(self):
        try:
            if not self.trade_history:
                return
                
            df = pd.DataFrame(self.trade_history)
            win_rate = len(df[df['profit_pct'] > 0]) / len(df) * 100
            avg_profit = df['profit_pct'].mean()
            total_profit = df['profit_pct'].sum()
            
            report = {
                'total_trades': len(df),
                'win_rate': win_rate,
                'average_profit': avg_profit,
                'total_profit': total_profit,
                'profit_by_direction': {
                    'LONG': df[df['direction'] == 'LONG']['profit_pct'].mean(),
                    'SHORT': df[df['direction'] == 'SHORT']['profit_pct'].mean()
                },
                'timeframe': self.timeframe,
                'period': f"{self.short_ma}/{self.long_ma} MA",
                'generated_at': datetime.now().isoformat()
            }
            
            report_file = f"logs/report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Reporte generado: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"Error al generar reporte: {e}")
            return None

    def plot_charts(self, data):
        try:
            plt.figure(figsize=(12, 8))
            
            # Gráfico de precios y medias móviles
            plt.subplot(2, 1, 1)
            plt.plot(data['timestamp'], data['close'], label='Precio', color='black', alpha=0.5)
            plt.plot(data['timestamp'], data['short_ma'], label=f'MA {self.short_ma}', color='blue')
            plt.plot(data['timestamp'], data['long_ma'], label=f'MA {self.long_ma}', color='red')
            plt.title(f'Análisis {self.symbol} - {self.timeframe}')
            plt.legend()
            plt.grid(True)
            
            # Gráfico de RSI
            plt.subplot(2, 1, 2)
            plt.plot(data['timestamp'], data['rsi'], label='RSI 14', color='purple')
            plt.axhline(70, color='red', linestyle='--')
            plt.axhline(30, color='green', linestyle='--')
            plt.legend()
            plt.grid(True)
            
            chart_file = f"charts/{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(chart_file)
            plt.close()
            
            return chart_file
            
        except Exception as e:
            logger.error(f"Error al generar gráficos: {e}")
            return None

    def run(self):
        logger.info("Iniciando bot de trading...")
        logger.info(f"Configuración: {config}")
        
        while True:
            try:
                now = datetime.now()
                
                # Verificar trades activos cada minuto
                self.check_take_profit_stop_loss()
                
                # Ejecutar análisis en cada nuevo período
                if self.last_check is None or (
                    (self.timeframe.endswith('m') and now.minute % int(self.timeframe[:-1]) == 0) or
                    (self.timeframe.endswith('h') and now.minute == 0 and now.hour % int(self.timeframe[:-1]) == 0) or
                    (self.timeframe.endswith('d') and now.hour == 0 and now.minute == 0)
                ):
                    self.last_check = now
                    
                    # Obtener datos y calcular indicadores
                    data = self.get_historical_data(limit=100)
                    if data is None:
                        continue
                        
                    data = self.calculate_indicators(data)
                    
                    # Generar y ejecutar señal
                    signal = self.generate_signal(data)
                    if signal != 'HOLD':
                        self.execute_trade(signal)
                    
                    # Generar gráficos y reportes
                    self.plot_charts(data)
                    if len(self.trade_history) > 0 and len(self.trade_history) % 5 == 0:
                        self.generate_report()
                    
                    logger.info(f"Análisis completado a las {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(60)  # Esperar 1 minuto entre verificaciones
                
            except KeyboardInterrupt:
                logger.info("Deteniendo bot...")
                self.generate_report()
                break
                
            except Exception as e:
                logger.error(f"Error en el bucle principal: {e}")
                time.sleep(60)

if __name__ == '__main__':
    bot = TradingBot()
    bot.run()
