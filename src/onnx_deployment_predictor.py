"""
Real-time Trade Cost Predictor - Deployment System (ONNX Optimized)
=================================================================

This system provides real-time trade cost predictions using ONNX models for faster inference.
It fetches live market data, calculates features, and recommends the best execution venue.

Features:
- Real-time market data fetching
- Live slippage prediction using ONNX models
- Best execution venue recommendation
- REST API for integration
- Performance monitoring
- Optimized inference with ONNX Runtime
"""

import numpy as np
import pandas as pd
import ccxt
import time
import json
import joblib
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from flask import Flask, request, jsonify
import logging

# ONNX Runtime for fast inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime not available - please install: pip install onnxruntime")
    exit(1)

@dataclass
class TradeRequest:
    """Structure for trade cost prediction requests"""
    symbol: str
    amount: float  # Amount in crypto
    side: str  # 'buy' or 'sell'
    amount_usd: Optional[float] = None

@dataclass
class ExchangeQuote:
    """Structure for exchange quotes with predicted costs"""
    exchange: str
    symbol: str
    quote_price: float
    predicted_slippage: float
    total_cost: float  # Including all fees and slippage
    recommendation_score: float
    fees: Dict[str, float]

class ONNXPredictor:
    """ONNX-based trade cost prediction system"""
    
    def __init__(self, model_path: str = None, model_type: str = 'lightgbm'):
        self.exchanges = {}
        self.model_session = None
        self.scaler_session = None
        self.feature_columns = []
        self.model_type = model_type
        self.metadata = {}
        
        # Initialize exchanges
        self.initialize_exchanges()
        
        # Load ONNX models
        self.load_onnx_models(model_path, model_type)
        
        # Exchange fees (you should update these with current rates)
        self.exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
            'coinbase': {'maker': 0.005, 'taker': 0.005},
            'okx': {'maker': 0.0008, 'taker': 0.001}
        }
    
    def initialize_exchanges(self):
        """Initialize exchange connections"""
        exchange_configs = {
            'binance': ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}}),
            'kraken': ccxt.kraken({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
            'okx': ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        }
        
        for name, exchange in exchange_configs.items():
            try:
                exchange.load_markets()
                self.exchanges[name] = exchange
                print(f"✅ Connected to {name}")
            except Exception as e:
                print(f"❌ Failed to connect to {name}: {e}")
    
    def load_onnx_models(self, model_path: str = None, model_type: str = 'lightgbm'):
        """Load ONNX models and metadata"""
        try:
            # Default to models directory if no path specified
            if model_path is None:
                model_dir = 'models'
            else:
                model_dir = model_path
            
            # Construct file paths - support fallback to model.onnx
            model_file = os.path.join(model_dir, f'{model_type}_model.onnx')
            if not os.path.exists(model_file):
                # Fallback to generic model.onnx
                model_file = os.path.join(model_dir, 'model.onnx')
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")
                print(f"⚠️ Using generic model.onnx (fallback)")
            
            scaler_file = os.path.join(model_dir, f'{model_type}_scaler.onnx')
            metadata_file = os.path.join(model_dir, f'{model_type}_metadata.json')
            features_json = os.path.join(model_dir, 'features.json')
            features_file = os.path.join(model_dir, 'feature_columns.pkl')
            
            # Load ONNX model
            print(f"🔄 Loading ONNX model: {model_file}")
            self.model_session = ort.InferenceSession(
                model_file,
                providers=['CPUExecutionProvider']  # Use CPU provider for compatibility
            )
            print(f"✅ Loaded ONNX model")
            
            # Load ONNX scaler (optional - skip if not available)
            self.scaler_session = None
            if os.path.exists(scaler_file):
                print(f"🔄 Loading ONNX scaler: {scaler_file}")
                self.scaler_session = ort.InferenceSession(
                    scaler_file,
                    providers=['CPUExecutionProvider']
                )
                print(f"✅ Loaded ONNX scaler")
            else:
                print(f"⚠️ Scaler file not found: {scaler_file} - skipping scaling")
            
            # Load metadata if available
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                print(f"✅ Loaded metadata")
            
            # Load feature columns - prioritize features.json
            if os.path.exists(features_json):
                with open(features_json, 'r') as f:
                    self.feature_columns = json.load(f)
                print(f"✅ Loaded feature columns from features.json ({len(self.feature_columns)} features)")
            elif os.path.exists(features_file):
                self.feature_columns = joblib.load(features_file)
                print(f"✅ Loaded feature columns from pickle ({len(self.feature_columns)} features)")
            elif 'feature_names' in self.metadata:
                self.feature_columns = self.metadata['feature_names']
                print(f"✅ Using feature columns from metadata ({len(self.feature_columns)} features)")
            else:
                raise FileNotFoundError("No feature columns found in features.json, feature_columns.pkl, or metadata")
            
            # Print model info
            model_inputs = self.model_session.get_inputs()
            model_outputs = self.model_session.get_outputs()
            print(f"📊 Model input shape: {model_inputs[0].shape}")
            print(f"📊 Model output shape: {model_outputs[0].shape}")
            
            if self.scaler_session:
                scaler_inputs = self.scaler_session.get_inputs()
                print(f"📊 Scaler input shape: {scaler_inputs[0].shape}")
            
        except Exception as e:
            print(f"❌ Failed to load ONNX models: {e}")
            raise
    
    def get_market_data(self, exchange_name: str, symbol: str) -> Optional[Dict]:
        """Fetch real-time market data"""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Get order book
            order_book = exchange.fetch_order_book(symbol, limit=20)
            
            # Get recent trades
            trades = exchange.fetch_trades(symbol, limit=50)
            
            # Get ticker for additional info
            ticker = exchange.fetch_ticker(symbol)
            
            return {
                'order_book': order_book,
                'trades': trades,
                'ticker': ticker,
                'timestamp': exchange.milliseconds()
            }
            
        except Exception as e:
            print(f"Error fetching data from {exchange_name}: {e}")
            return None
    
    def calculate_features(self, market_data: Dict, trade_request: TradeRequest) -> Optional[Dict]:
        """Calculate features for the ML model"""
        try:
            order_book = market_data['order_book']
            trades = market_data['trades']
            
            if not order_book['bids'] or not order_book['asks']:
                return None
            
            # Basic price information
            best_bid = order_book['bids'][0][0]
            best_ask = order_book['asks'][0][0]
            mid_price = (best_bid + best_ask) / 2
            
            # Convert trade side to numeric
            trade_side = 1 if trade_request.side.lower() == 'buy' else 0
            
            # Calculate order size in USD if not provided
            if trade_request.amount_usd is None:
                order_size_usd = trade_request.amount * mid_price
            else:
                order_size_usd = trade_request.amount_usd
            
            # Spread metrics
            spread = best_ask - best_bid
            spread_percentage = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            # Market depth calculations
            market_depth_level_1 = (order_book['bids'][0][1] * best_bid + 
                                   order_book['asks'][0][1] * best_ask)
            
            # Level 5 depth
            bid_depth_5 = sum([bid[1] * bid[0] for bid in order_book['bids'][:5]])
            ask_depth_5 = sum([ask[1] * ask[0] for ask in order_book['asks'][:5]])
            market_depth_level_5 = bid_depth_5 + ask_depth_5
            
            # Level 10 depth
            bid_depth_10 = sum([bid[1] * bid[0] for bid in order_book['bids'][:10]])
            ask_depth_10 = sum([ask[1] * ask[0] for ask in order_book['asks'][:10]])
            market_depth_level_10 = bid_depth_10 + ask_depth_10
            
            # Percentage-based depth (0.5% from mid-price)
            upper_bound = mid_price * 1.005
            lower_bound = mid_price * 0.995
            
            bid_depth_pct = sum([bid[1] * bid[0] for bid in order_book['bids'] if bid[0] >= lower_bound])
            ask_depth_pct = sum([ask[1] * ask[0] for ask in order_book['asks'] if ask[0] <= upper_bound])
            market_depth_percent_05 = bid_depth_pct + ask_depth_pct
            
            # Order book imbalance
            total_bids = sum([bid[1] for bid in order_book['bids'][:10]])
            total_asks = sum([ask[1] for ask in order_book['asks'][:10]])
            total_volume = total_bids + total_asks
            order_book_imbalance = total_bids / total_volume if total_volume > 0 else 0.5
            
            # Price slopes
            ask_prices = [ask[0] for ask in order_book['asks'][:5]]
            bid_prices = [bid[0] for bid in order_book['bids'][:5]]
            
            ask_price_slope = (ask_prices[-1] - ask_prices[0]) / ask_prices[0] * 100 if len(ask_prices) >= 5 else 0
            bid_price_slope = (bid_prices[0] - bid_prices[-1]) / bid_prices[0] * 100 if len(bid_prices) >= 5 else 0
            
            # Trade volatility and volume (last 1 minute)
            current_time = time.time() * 1000
            recent_trades = [t for t in trades if current_time - t['timestamp'] <= 60000]
            
            if recent_trades:
                prices = [t['price'] for t in recent_trades]
                volumes = [t['amount'] for t in recent_trades]
                trade_volatility_1m = np.std(prices) if len(prices) > 1 else 0
                trade_volume_1m = sum(volumes)
            else:
                trade_volatility_1m = 0
                trade_volume_1m = 0
            
            # Calculate depth utilization (important feature from training)
            if market_depth_level_10 > 0:
                depth_utilization = order_size_usd / market_depth_level_10
            else:
                depth_utilization = 0
            
            # Additional features
            relative_order_size = order_size_usd / market_depth_level_10 if market_depth_level_10 > 0 else 0
            bid_ask_ratio = len(order_book['bids']) / max(len(order_book['asks']), 1)
            order_book_depth_ratio = bid_depth_5 / max(ask_depth_5, 1) if ask_depth_5 > 0 else 1
            
            # Base features dictionary
            features = {
                'order_size_fiat': order_size_usd,
                'order_size_crypto': trade_request.amount,
                'trade_side': trade_side,
                'spread_percentage': spread_percentage,
                'market_depth_level_1': market_depth_level_1,
                'market_depth_level_5': market_depth_level_5,
                'market_depth_level_10': market_depth_level_10,
                'market_depth_percent_0.5': market_depth_percent_05,
                'market_depth_percent_1.0': market_depth_percent_05 * 2,  # Approximation for 1%
                'order_book_imbalance': order_book_imbalance,
                'ask_price_slope': ask_price_slope,
                'bid_price_slope': bid_price_slope,
                'trade_volatility_1m': trade_volatility_1m,
                'trade_volume_1m': trade_volume_1m,
                'mid_price': mid_price,
                'relative_order_size': relative_order_size,
                'bid_ask_ratio': bid_ask_ratio,
                'order_book_depth_ratio': order_book_depth_ratio,
                'depth_utilization': depth_utilization  # Important feature from training
            }
            
            # Add engineered features (matching training pipeline)
            self._add_engineered_features(features)
            
            return features
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return None
    
    def _add_engineered_features(self, features: Dict):
        """Add engineered features that were created during training"""
        # Interaction features
        features['size_spread_interaction'] = features['order_size_fiat'] * features['spread_percentage']
        features['size_volatility_interaction'] = features['order_size_fiat'] * features['trade_volatility_1m']
        features['depth_utilization_squared'] = features['depth_utilization'] ** 2
        
        features['imbalance_spread'] = features['order_book_imbalance'] * features['spread_percentage']
        features['depth_spread_ratio'] = features['market_depth_level_5'] / (features['spread_percentage'] + 1e-8)
        features['volatility_spread'] = features['trade_volatility_1m'] * features['spread_percentage']
        
        features['avg_price_slope'] = (features['ask_price_slope'] + features['bid_price_slope']) / 2
        features['price_slope_asymmetry'] = features['ask_price_slope'] - features['bid_price_slope']
        
        features['depth_ratio_1_5'] = features['market_depth_level_1'] / (features['market_depth_level_5'] + 1e-8)
        features['depth_ratio_5_10'] = features['market_depth_level_5'] / (features['market_depth_level_10'] + 1e-8)
        
        # Logarithmic features
        log_features = ['order_size_fiat', 'market_depth_level_1', 'market_depth_level_5', 
                       'market_depth_level_10', 'trade_volume_1m']
        
        for feature in log_features:
            if feature in features:
                features[f'{feature}_log'] = np.log1p(features[feature])
    
    def predict_slippage(self, features: Dict) -> float:
        """Predict slippage using ONNX models"""
        try:
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            # Convert to numpy array with correct shape and type
            input_array = np.array([feature_vector], dtype=np.float32)
            
            # Get input name for ONNX model
            model_input_name = self.model_session.get_inputs()[0].name
            
            # Apply scaling if scaler is available, otherwise skip
            if self.scaler_session is not None:
                scaler_input_name = self.scaler_session.get_inputs()[0].name
                scaled_features = self.scaler_session.run(
                    None, 
                    {scaler_input_name: input_array}
                )[0]
            else:
                # Skip scaling - use features directly
                scaled_features = input_array
            
            # Make prediction using ONNX model
            prediction = self.model_session.run(
                None,
                {model_input_name: scaled_features}
            )[0]
            
            # Extract scalar prediction
            predicted_slippage = float(prediction[0][0] if prediction.ndim > 1 else prediction[0])
            
            # Ensure reasonable bounds (slippage should be positive and reasonable)
            predicted_slippage = max(0.0001, min(predicted_slippage, 0.1))  # Between 0.01% and 10%
            
            return predicted_slippage
            
        except Exception as e:
            print(f"Error predicting slippage: {e}")
            return 0.01  # Return a conservative default (1%)
    
    def calculate_total_cost(self, quote_price: float, predicted_slippage: float, 
                           exchange_name: str, trade_side: str, amount: float) -> Tuple[float, Dict]:
        """Calculate total execution cost including all fees"""
        
        # Get exchange fees
        fees = self.exchange_fees.get(exchange_name, {'maker': 0.001, 'taker': 0.001})
        trading_fee = fees['taker']  # Assume market orders (taker fees)
        
        # Calculate slipped price
        if trade_side.lower() == 'buy':
            actual_price = quote_price * (1 + predicted_slippage)
        else:
            actual_price = quote_price * (1 - predicted_slippage)
        
        # Calculate costs
        gross_cost = amount * actual_price
        fee_cost = gross_cost * trading_fee
        total_cost = gross_cost + fee_cost
        
        fee_breakdown = {
            'trading_fee': fee_cost,
            'slippage_cost': amount * abs(actual_price - quote_price),
            'total_fees': fee_cost
        }
        
        return total_cost, fee_breakdown
    
    async def get_best_execution_venue(self, trade_request: TradeRequest) -> List[ExchangeQuote]:
        """Find the best execution venue for a trade"""
        quotes = []
        
        for exchange_name in self.exchanges.keys():
            try:
                # Get market data
                market_data = self.get_market_data(exchange_name, trade_request.symbol)
                if not market_data:
                    continue
                
                # Calculate features
                features = self.calculate_features(market_data, trade_request)
                if not features:
                    continue
                
                # Predict slippage using ONNX
                predicted_slippage = self.predict_slippage(features)
                
                # Get quote price
                order_book = market_data['order_book']
                if trade_request.side.lower() == 'buy':
                    quote_price = order_book['asks'][0][0]
                else:
                    quote_price = order_book['bids'][0][0]
                
                # Calculate total cost
                total_cost, fees = self.calculate_total_cost(
                    quote_price, predicted_slippage, exchange_name, 
                    trade_request.side, trade_request.amount
                )
                
                # Calculate recommendation score (lower cost = higher score)
                recommendation_score = 1.0 / (total_cost + 1e-8)
                
                quote = ExchangeQuote(
                    exchange=exchange_name,
                    symbol=trade_request.symbol,
                    quote_price=quote_price,
                    predicted_slippage=predicted_slippage,
                    total_cost=total_cost,
                    recommendation_score=recommendation_score,
                    fees=fees
                )
                
                quotes.append(quote)
                
            except Exception as e:
                print(f"Error processing {exchange_name}: {e}")
                continue
        
        # Sort by total cost (ascending)
        quotes.sort(key=lambda x: x.total_cost)
        
        return quotes

class TradeCostAPI:
    """REST API for trade cost predictions using ONNX models"""
    
    def __init__(self, predictor: ONNXPredictor):
        self.predictor = predictor
        self.app = Flask(__name__)
        self.setup_routes()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_routes(self):
        """Setup API routes"""

        @self.app.route('/', methods=['GET'])
        def home():
            """Home page with API documentation"""
            return jsonify({
                'service': 'AI Trade Cost Predictor API (ONNX Optimized)',
                'version': '2.0.0',
                'status': 'active',
                'model_type': self.predictor.model_type,
                'inference_engine': 'ONNX Runtime',
                'performance': self.predictor.metadata.get('model_performance', {}),
                'endpoints': {
                    'GET /health': 'Health check',
                    'POST /predict': 'Get trade cost predictions for all exchanges',
                    'POST /compare': 'Compare specific exchanges',
                    'GET /models': 'Available model information'
                },
                'example_request': {
                    'symbol': 'BTC/USDT',
                    'amount': 1.0,
                    'side': 'sell'
                }
            })
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'exchanges': list(self.predictor.exchanges.keys()),
                'model_type': self.predictor.model_type,
                'inference_engine': 'ONNX Runtime',
                'feature_count': len(self.predictor.feature_columns),
                'model_performance': self.predictor.metadata.get('model_performance', {})
            })
        
        @self.app.route('/models', methods=['GET'])
        def models():
            """Get available model information"""
            available_models = []
            model_dir = 'models'
            
            for model_type in ['lightgbm', 'random_forest', 'xgboost']:
                model_file = os.path.join(model_dir, f'{model_type}_model.onnx')
                scaler_file = os.path.join(model_dir, f'{model_type}_scaler.onnx')
                metadata_file = os.path.join(model_dir, f'{model_type}_metadata.json')
                
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    model_info = {'type': model_type, 'status': 'available'}
                    
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            model_info['performance'] = metadata.get('model_performance', {})
                            model_info['training_date'] = metadata.get('training_date', 'N/A')
                        except:
                            pass
                    
                    available_models.append(model_info)
            
            return jsonify({
                'current_model': self.predictor.model_type,
                'available_models': available_models
            })
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Main prediction endpoint"""
            try:
                data = request.json
                
                # Validate request
                if not all(key in data for key in ['symbol', 'amount', 'side']):
                    return jsonify({'error': 'Missing required fields: symbol, amount, side'}), 400
                
                # Create trade request
                trade_request = TradeRequest(
                    symbol=data['symbol'],
                    amount=float(data['amount']),
                    side=data['side'],
                    amount_usd=data.get('amount_usd')
                )
                
                # Get predictions
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                quotes = loop.run_until_complete(
                    self.predictor.get_best_execution_venue(trade_request)
                )
                
                if not quotes:
                    return jsonify({'error': 'No quotes available'}), 500
                
                # Format response
                response = {
                    'request': {
                        'symbol': trade_request.symbol,
                        'amount': trade_request.amount,
                        'side': trade_request.side
                    },
                    'model_info': {
                        'type': self.predictor.model_type,
                        'inference_engine': 'ONNX Runtime',
                        'performance': self.predictor.metadata.get('model_performance', {})
                    },
                    'timestamp': datetime.now().isoformat(),
                    'best_venue': quotes[0].exchange,
                    'potential_savings': quotes[-1].total_cost - quotes[0].total_cost if len(quotes) > 1 else 0,
                    'quotes': []
                }
                
                for i, quote in enumerate(quotes):
                    quote_data = {
                        'rank': i + 1,
                        'exchange': quote.exchange,
                        'quote_price': quote.quote_price,
                        'predicted_slippage': quote.predicted_slippage,
                        'predicted_slippage_pct': quote.predicted_slippage * 100,
                        'total_cost': quote.total_cost,
                        'fees': quote.fees,
                        'savings_vs_worst': quotes[-1].total_cost - quote.total_cost if len(quotes) > 1 else 0,
                        'cost_rank': f"{i+1}/{len(quotes)}"
                    }
                    response['quotes'].append(quote_data)
                
                self.logger.info(f"ONNX Prediction: {trade_request.symbol} {trade_request.amount} {trade_request.side} - Best: {quotes[0].exchange}")
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/compare', methods=['POST'])
        def compare_exchanges():
            """Compare specific exchanges"""
            try:
                data = request.json
                trade_request = TradeRequest(
                    symbol=data['symbol'],
                    amount=float(data['amount']),
                    side=data['side'],
                    amount_usd=data.get('amount_usd')
                )
                
                exchanges = data.get('exchanges', list(self.predictor.exchanges.keys()))
                
                # Validate requested exchanges exist
                valid_exchanges = [ex for ex in exchanges if ex in self.predictor.exchanges]
                if not valid_exchanges:
                    return jsonify({'error': 'No valid exchanges specified'}), 400
                
                # Filter predictor exchanges temporarily
                original_exchanges = self.predictor.exchanges.copy()
                self.predictor.exchanges = {k: v for k, v in original_exchanges.items() if k in valid_exchanges}
                
                # Get quotes
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                quotes = loop.run_until_complete(
                    self.predictor.get_best_execution_venue(trade_request)
                )
                
                # Restore original exchanges
                self.predictor.exchanges = original_exchanges
                
                if not quotes:
                    return jsonify({'error': 'No quotes available for specified exchanges'}), 500
                
                # Calculate comparison metrics
                best_cost = min(quote.total_cost for quote in quotes)
                worst_cost = max(quote.total_cost for quote in quotes)
                potential_savings = worst_cost - best_cost
                
                response = {
                    'comparison_summary': {
                        'exchanges_compared': [q.exchange for q in quotes],
                        'best_exchange': quotes[0].exchange,
                        'best_cost': best_cost,
                        'worst_cost': worst_cost,
                        'potential_savings': potential_savings,
                        'savings_percentage': (potential_savings / worst_cost) * 100 if worst_cost > 0 else 0,
                        'model_type': self.predictor.model_type
                    },
                    'detailed_quotes': [
                        {
                            'exchange': q.exchange,
                            'total_cost': q.total_cost,
                            'predicted_slippage_pct': q.predicted_slippage * 100,
                            'quote_price': q.quote_price,
                            'fees': q.fees,
                            'cost_difference_from_best': q.total_cost - best_cost
                        } for q in quotes
                    ],
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Comparison error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/switch_model', methods=['POST'])
        def switch_model():
            """Switch to a different ONNX model"""
            try:
                data = request.json
                new_model_type = data.get('model_type', 'lightgbm')
                
                if new_model_type not in ['lightgbm', 'random_forest', 'xgboost']:
                    return jsonify({'error': 'Invalid model type. Use: lightgbm, random_forest, or xgboost'}), 400
                
                # Try to load new model
                old_model_type = self.predictor.model_type
                try:
                    self.predictor.load_onnx_models(model_type=new_model_type)
                    self.predictor.model_type = new_model_type
                    
                    return jsonify({
                        'message': f'Successfully switched from {old_model_type} to {new_model_type}',
                        'new_model_performance': self.predictor.metadata.get('model_performance', {}),
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as load_error:
                    return jsonify({
                        'error': f'Failed to load {new_model_type} model: {str(load_error)}',
                        'current_model': old_model_type
                    }), 500
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=None, debug=False):
        """Run the API server"""
        # Read port from environment variable (Railway) or use default
        if port is None:
            port = int(os.environ.get('PORT', 5000))
        
        print(f"🚀 Starting ONNX-powered Trade Cost Prediction API on {host}:{port}")
        print(f"🤖 Using model: {self.predictor.model_type}")
        self.app.run(host=host, port=port, debug=debug)

# Utility functions for model switching and testing
def list_available_models(model_dir: str = 'models') -> List[str]:
    """List all available ONNX models"""
    available = []
    for model_type in ['lightgbm', 'random_forest', 'xgboost']:
        model_file = os.path.join(model_dir, f'{model_type}_model.onnx')
        scaler_file = os.path.join(model_dir, f'{model_type}_scaler.onnx')
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            available.append(model_type)
    
    return available

def benchmark_models(model_dir: str = 'models', test_symbol: str = 'BTC/USDT', 
                    test_amount: float = 1.0, iterations: int = 10):
    """Benchmark different ONNX models for speed and compare predictions"""
    print("🏁 Benchmarking ONNX Models")
    print("="*50)
    
    available_models = list_available_models(model_dir)
    if not available_models:
        print("❌ No ONNX models found")
        return
    
    results = {}
    
    for model_type in available_models:
        print(f"\n🔄 Testing {model_type}...")
        
        try:
            # Initialize predictor
            predictor = ONNXPredictor(model_path=model_dir, model_type=model_type)
            
            # Create test request
            trade_request = TradeRequest(
                symbol=test_symbol,
                amount=test_amount,
                side='sell'
            )
            
            # Time the predictions
            start_time = time.time()
            
            predictions = []
            for i in range(iterations):
                # Get market data (use first available exchange)
                exchange_name = list(predictor.exchanges.keys())[0]
                market_data = predictor.get_market_data(exchange_name, test_symbol)
                
                if market_data:
                    features = predictor.calculate_features(market_data, trade_request)
                    if features:
                        prediction = predictor.predict_slippage(features)
                        predictions.append(prediction)
            
            end_time = time.time()
            
            if predictions:
                avg_time = (end_time - start_time) / len(predictions)
                avg_prediction = np.mean(predictions)
                std_prediction = np.std(predictions)
                
                results[model_type] = {
                    'avg_inference_time_ms': avg_time * 1000,
                    'avg_prediction': avg_prediction,
                    'prediction_std': std_prediction,
                    'predictions_per_second': 1 / avg_time,
                    'successful_predictions': len(predictions)
                }
                
                print(f"✅ {model_type}:")
                print(f"   - Avg inference time: {avg_time*1000:.2f}ms")
                print(f"   - Predictions/second: {1/avg_time:.1f}")
                print(f"   - Avg slippage prediction: {avg_prediction*100:.4f}%")
                print(f"   - Prediction std: {std_prediction*100:.4f}%")
            else:
                print(f"❌ {model_type}: No successful predictions")
                
        except Exception as e:
            print(f"❌ {model_type}: Error - {e}")
    
    # Print summary
    if results:
        print(f"\n🏆 Benchmark Summary:")
        fastest = min(results.items(), key=lambda x: x[1]['avg_inference_time_ms'])
        print(f"   - Fastest model: {fastest[0]} ({fastest[1]['avg_inference_time_ms']:.2f}ms)")
        
        most_throughput = max(results.items(), key=lambda x: x[1]['predictions_per_second'])
        print(f"   - Highest throughput: {most_throughput[0]} ({most_throughput[1]['predictions_per_second']:.1f} pred/sec)")

# Example usage and testing
def test_onnx_predictor(model_type: str = 'lightgbm'):
    """Test the ONNX prediction system"""
    print(f"🧪 Testing ONNX Trade Cost Predictor - {model_type}")
    print("="*60)
    
    # Check available models first
    available = list_available_models()
    print(f"📦 Available ONNX models: {available}")
    
    if model_type not in available:
        print(f"❌ Model {model_type} not available. Using {available[0]}")
        model_type = available[0]
    
    # Initialize predictor with ONNX models
    predictor = ONNXPredictor(model_path='models', model_type=model_type)
    
    # Test trade request
    trade_request = TradeRequest(
        symbol='BTC/USDT',
        amount=1.0,  # 1 BTC
        side='sell'
    )
    
    print(f"📊 Testing trade: {trade_request.side} {trade_request.amount} {trade_request.symbol}")
    print(f"🤖 Using model: {model_type} (ONNX)")
    
    # Time the prediction
    start_time = time.time()
    
    # Get quotes
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    quotes = loop.run_until_complete(predictor.get_best_execution_venue(trade_request))
    
    end_time = time.time()
    
    if quotes:
        print(f"\n⚡ Prediction completed in {(end_time - start_time)*1000:.1f}ms")
        print(f"🏆 Best execution venue: {quotes[0].exchange}")
        print(f"💰 Predicted total cost: ${quotes[0].total_cost:,.2f}")
        print(f"📉 Predicted slippage: {quotes[0].predicted_slippage*100:.4f}%")
        
        if len(quotes) > 1:
            savings = quotes[-1].total_cost - quotes[0].total_cost
            savings_pct = (savings / quotes[-1].total_cost) * 100
            print(f"💡 Potential savings: ${savings:,.2f} ({savings_pct:.2f}%)")
        
        print(f"\n📋 All quotes:")
        for i, quote in enumerate(quotes):
            savings = quotes[-1].total_cost - quote.total_cost if len(quotes) > 1 else 0
            print(f"{i+1}. {quote.exchange}: ${quote.total_cost:,.2f} "
                  f"(slippage: {quote.predicted_slippage*100:.4f}%, "
                  f"savings: ${savings:,.2f})")
    else:
        print("❌ No quotes received")

def main():
    """Main function to run the ONNX-optimized system"""
    print("🚀 AI-Powered Trade Execution Cost Predictor - ONNX Deployment")
    print("="*70)
    
    if not ONNX_AVAILABLE:
        print("❌ ONNX Runtime not available. Please install it:")
        print("   pip install onnxruntime")
        return
    
    try:
        # Check for model.onnx (generic) or typed models
        model_dir = 'models'
        generic_model = os.path.join(model_dir, 'model.onnx')
        
        if os.path.exists(generic_model):
            # Use generic model.onnx
            model_type = 'lightgbm'  # Default type for generic model
            print(f"📦 Found generic model.onnx")
        else:
            # Check available typed models
            available_models = list_available_models(model_dir)
            if not available_models:
                print("❌ No ONNX models found in 'models' directory")
                print("Make sure you have model.onnx or {model_type}_model.onnx")
                return
            
            print(f"📦 Available ONNX models: {available_models}")
            model_type = 'lightgbm' if 'lightgbm' in available_models else available_models[0]
        
        print(f"🤖 Using model: {model_type}")
        
        # Initialize ONNX predictor
        predictor = ONNXPredictor(model_path=model_dir, model_type=model_type)
        
        # Skip test in production (comment out for Railway)
        # print(f"\n🧪 Running test predictions...")
        # test_onnx_predictor(model_type)
        
        # Start API server
        api = TradeCostAPI(predictor)
        
        print(f"\n🌐 Starting ONNX-powered REST API...")
        print("Endpoints:")
        print("  GET  /health - Health check and model info")
        print("  GET  /models - Available models information")
        print("  POST /predict - Get trade cost predictions")
        print("  POST /compare - Compare specific exchanges")
        print(f"\nCurrent model: {model_type} (ONNX)")
        print(f"Features: {len(predictor.feature_columns)}")
        
        print("\nExample request to /predict:")
        print("""{
  "symbol": "BTC/USDT",
  "amount": 1.0,
  "side": "sell"
}""")
        
        # Run API server (port will be read from PORT env var)
        api.run(debug=False)
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure model.onnx exists in 'models' directory")
        print("2. Check that features.json exists")
        print("3. Verify ONNX Runtime is installed: pip install onnxruntime")

# Additional utility functions for ONNX model management
def validate_onnx_models(model_dir: str = 'models'):
    """Validate that all ONNX models are working correctly"""
    print("🔍 Validating ONNX Models")
    print("="*40)
    
    available_models = list_available_models(model_dir)
    
    for model_type in available_models:
        print(f"\n🔄 Validating {model_type}...")
        
        try:
            # Try to load the model
            predictor = ONNXPredictor(model_path=model_dir, model_type=model_type)
            
            # Create dummy features
            dummy_features = {col: 1.0 for col in predictor.feature_columns}
            
            # Test prediction
            prediction = predictor.predict_slippage(dummy_features)
            
            print(f"✅ {model_type}: Valid (test prediction: {prediction*100:.4f}%)")
            
            if predictor.metadata:
                perf = predictor.metadata.get('model_performance', {})
                if perf:
                    print(f"   - R² Score: {perf.get('r2_score', 'N/A')}")
                    print(f"   - MAE: {perf.get('mae', 'N/A')}")
            
        except Exception as e:
            print(f"❌ {model_type}: Error - {e}")

def create_model_comparison_report(model_dir: str = 'models'):
    """Create a detailed comparison report of all ONNX models"""
    print("📊 ONNX Model Comparison Report")
    print("="*50)
    
    available_models = list_available_models(model_dir)
    
    for model_type in available_models:
        metadata_file = os.path.join(model_dir, f'{model_type}_metadata.json')
        
        print(f"\n📈 {model_type.upper()} Model:")
        print("-" * 30)
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            perf = metadata.get('model_performance', {})
            if perf:
                print(f"R² Score: {perf.get('r2_score', 'N/A'):.6f}")
                print(f"MAE: {perf.get('mae', 'N/A'):.6f}")
                print(f"RMSE: {perf.get('rmse', 'N/A'):.6f}")
                print(f"Mean Absolute Error %: {perf.get('mae_percentage', 'N/A'):.4f}%")
            
            print(f"Training Date: {metadata.get('training_date', 'N/A')}")
            print(f"Features: {len(metadata.get('feature_names', []))}")
        else:
            print("No metadata available")

if __name__ == "__main__":
    import sys
    
    # Command line options
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'validate':
            validate_onnx_models()
        elif command == 'benchmark':
            benchmark_models(iterations=20)
        elif command == 'compare':
            create_model_comparison_report()
        elif command == 'test':
            model_type = sys.argv[2] if len(sys.argv) > 2 else 'lightgbm'
            test_onnx_predictor(model_type)
        else:
            print("Available commands: validate, benchmark, compare, test [model_type]")
    else:
        main()