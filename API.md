# API Usage Guide

## Base URL
```
https://web-production-97230.up.railway.app
```

## Quick Start

### 1. Health Check
```bash
curl https://web-production-97230.up.railway.app/health
```

### 2. Get Predictions
```bash
curl -X POST https://web-production-97230.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT",
    "amount": 1.0,
    "side": "sell"
  }'
```

### 3. Compare Exchanges
```bash
curl -X POST https://web-production-97230.up.railway.app/compare \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETH/USDT",
    "amount": 10.0,
    "side": "buy",
    "exchanges": ["binance", "kraken"]
  }'
```

## Request Format

**Required fields:**
- `symbol`: Trading pair (e.g., "BTC/USDT", "ETH/USDT")
- `amount`: Amount in cryptocurrency
- `side`: "buy" or "sell"

**Optional fields:**
- `amount_usd`: Amount in USD (auto-calculated if not provided)

## Response Example

```json
{
  "best_venue": "binance",
  "quotes": [
    {
      "exchange": "binance",
      "quote_price": 43250.50,
      "predicted_slippage_pct": 0.12,
      "total_cost": 43250.50,
      "fees": {
        "trading_fee": 43.25,
        "slippage_cost": 51.90
      }
    }
  ]
}
```

## Code Examples

### JavaScript
```javascript
const response = await fetch('https://web-production-97230.up.railway.app/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    symbol: 'BTC/USDT',
    amount: 1.0,
    side: 'sell'
  })
});

const data = await response.json();
console.log('Best exchange:', data.best_venue);
```

### Python
```python
import requests

response = requests.post(
    'https://web-production-97230.up.railway.app/predict',
    json={'symbol': 'BTC/USDT', 'amount': 1.0, 'side': 'sell'}
)

data = response.json()
print(f"Best venue: {data['best_venue']}")
print(f"Total cost: ${data['quotes'][0]['total_cost']:,.2f}")
```

## Available Endpoints

- `GET /health` - Check API status
- `GET /` - API documentation
- `POST /predict` - Get predictions for all exchanges
- `POST /compare` - Compare specific exchanges

## Supported Exchanges

- Binance
- Kraken
- Coinbase
- OKX

