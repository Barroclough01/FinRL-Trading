<div align="center">
<img align="center" width="30%" alt="image" src="https://github.com/AI4Finance-Foundation/FinGPT/assets/31713746/e0371951-1ce1-488e-aa25-0992dafcc139">
</div>

# FinRL Trading Platform v2.0

![Visitors](https://api.visitorbadge.io/api/VisitorHit?user=AI4Finance-Foundation&repo=FinRL-Trading&countColor=%23B17A)

**A modern, modular quantitative trading platform built with Python, featuring machine learning strategies, comprehensive backtesting, and live trading capabilities.**

## 🚀 Key Features

- **📊 Multi-Source Data Pipeline**: Automated data acquisition from Yahoo Finance, FMP, and WRDS
- **🔄 Smart Data Source Selection**: Automatically selects best available data source (FMP > WRDS > Yahoo)
- **🤖 Strategy Framework**: Multiple quantitative strategies including ML-based stock selection
- **📈 Risk Management**: Comprehensive risk controls and position limits
- **💰 Live Trading**: Alpaca integration with paper and live trading support
- **🌐 Interactive Dashboard**: Streamlit-based web interface for monitoring and control
- **🐳 Production Ready**: Docker containerization for easy deployment
- **🔧 Modular Design**: Clean, extensible architecture following best practices

## 🏗️ Project Architecture

```
finrl-trading/
├── src/
│   ├── config/           # Centralized configuration management
│   │   └── settings.py   # Pydantic-based settings with environment variables
│   ├── data/            # Data acquisition and processing
│   │   ├── wrds_fetcher.py     # WRDS database integration
│   │   ├── data_processor.py   # Data cleaning and feature engineering
│   │   └── data_store.py       # SQLite-based data persistence
│   ├── backtest/      # backtesting
│   │   └── backtest_engine.py  # backtesting engine
│   ├── strategies/      # Trading strategies
│   │   ├── base_strategy.py    # Abstract strategy framework
│   │   └── ml_strategy.py      # ML-based stock selection
│   ├── trading/         # Live trading execution
│   │   ├── alpaca_manager.py   # Alpaca API integration
│   │   └── trade_executor.py   # Order execution and risk management
│   ├── web/            # Web interface
│   │   ├── app.py              # Main Streamlit application
│   │   └── components.py       # Reusable UI components
│   └── main.py         # CLI entry point
├── data/               # Runtime data storage (gitignored)
├── logs/               # Application logs (gitignored)
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-container setup
├── requirements.txt    # Python dependencies
└── setup.py           # Package installation
```

## 🛠️ Installation & Setup

### Prerequisites

- **Python 3.11+**
- **Docker** (optional, for containerized deployment)
- **Alpaca Account** (for live trading)
- **WRDS Access** (optional, for academic data)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/FinRL-Trading.git
   cd FinRL-Trading
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Copy template and edit
   cp .env.example .env
   # Edit .env with your API keys and settings
   nano .env  # or your preferred editor
   ```

4. **Launch the web dashboard**
   ```bash
   streamlit run src/web/app.py
   ```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run with Docker directly
docker build -t finrl-trading .
docker run -p 8501:8501 finrl-trading
```

### Examples & Tutorials

Check out the `examples/` folder for comprehensive tutorials:

```bash
# Complete interactive tutorial (recommended)
jupyter notebook examples/complete_trading_workflow.ipynb

# Quick start examples
python examples/simple_example.py
python examples/data_fetching_demo.py

# View all examples
cat examples/README.md
```

### CLI Usage

```bash
# Use the main CLI entry point
python src/main.py --help

# Available commands:
python src/main.py dashboard    # Start web dashboard
python src/main.py backtest     # Run strategy backtest
python src/main.py trade        # Execute live trading
python src/main.py data         # Manage data operations

# Or use installed package commands:
finrl dashboard                 # Start dashboard
finrl-backtest                  # Run backtest
finrl-trade                     # Execute trades
```

## 📖 Usage Examples

### Data Acquisition

```python
from src.data.wrds_fetcher import WRDSFetcher

# Initialize WRDS fetcher
fetcher = WRDSFetcher()

# Get S&P 500 components
components = fetcher.get_sp500_components()

# Fetch fundamental data
tickers = ['AAPL', 'MSFT', 'GOOGL']
fundamentals = fetcher.get_fundamental_data(
    tickers, '2020-01-01', '2023-12-31'
)

# Fetch price data
prices = fetcher.get_price_data(
    tickers, '2020-01-01', '2023-12-31'
)
```

### Strategy Development

```python
from src.strategies.base_strategy import create_strategy, StrategyConfig
from src.strategies.backtest_engine import BacktestEngine, BacktestConfig

# Create ML-based stock selection strategy
config = StrategyConfig(
    name="ML Stock Selector",
    parameters={'model_type': 'random_forest'},
    risk_limits={'max_weight': 0.1}
)

strategy = create_strategy("ml_strategy", config)

# Run backtest
backtest_config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=1000000
)

engine = BacktestEngine(backtest_config)
result = engine.run_backtest(strategy, price_data, weight_signals)

# View results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
```

### Data Source Usage

```python
from src.data.wrds_fetcher import get_data_manager

# Initialize data manager (automatically selects best available source)
manager = get_data_manager()

# Check which data source is being used
info = manager.get_source_info()
print(f"Using data source: {info['current_source']}")
print(f"Available sources: {info['available_sources']}")

# Fetch data (automatically uses best available source)
tickers = manager.get_unique_tickers(manager.get_sp500_components())
fundamentals = manager.get_fundamental_data(
    tickers[:50], '2020-01-01', '2023-12-31'
)
prices = manager.get_price_data(
    tickers[:50], '2020-01-01', '2023-12-31'
)
```

### Live Trading

```python
from src.trading.alpaca_manager import create_alpaca_account_from_env
from src.trading.trade_executor import TradeExecutor, ExecutionConfig

# Setup Alpaca connection
account = create_alpaca_account_from_env()
alpaca_manager = AlpacaManager([account])

# Configure execution settings
exec_config = ExecutionConfig(
    max_order_value=100000,
    risk_checks_enabled=True
)

executor = TradeExecutor(alpaca_manager, exec_config)

# Execute portfolio rebalance
target_weights = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}
result = executor.execute_portfolio_rebalance(target_weights)

print(f"Orders placed: {len(result.orders_placed)}")
```

## 🎯 Core Components

### Data Layer (`src/data/`)
- **Multi-Source Data Manager** (`wrds_fetcher.py`): Intelligent data source selection and management
- **Yahoo Finance Fetcher**: Free financial data from Yahoo Finance
- **FMP Fetcher**: Premium data from Financial Modeling Prep (API required)
- **WRDS Fetcher**: Academic database access (credentials required)
- **Data Processing** (`data_processor.py`): Feature engineering, cleaning, and quality checks
- **Data Persistence** (`data_store.py`): SQLite-based storage with caching and versioning

### Strategy Framework (`src/strategies/`)
- **Base Strategy** (`base_strategy.py`): Abstract framework for custom strategies
- **ML Strategies** (`ml_strategy.py`): Random Forest, Gradient Boosting for stock selection
- **Backtesting Engine** (`backtest_engine.py`): Comprehensive performance and risk analysis

**Available Strategies:**
- Equal Weight Strategy
- Market Cap Weighted Strategy
- ML-based Stock Selection
- Sector Neutral ML Strategy

### Trading System (`src/trading/`)
- **Alpaca Integration** (`alpaca_manager.py`): API client with multi-account support
- **Trade Execution** (`trade_executor.py`): Order management with risk controls
- **Real-time Monitoring**: Position tracking and P&L calculation

### Web Dashboard (`src/web/`)
- **Streamlit Interface** (`app.py`): Interactive web application
- **Data Visualization** (`components.py`): Reusable charting components
- **Live Monitoring**: Real-time portfolio and strategy performance

### Configuration System (`src/config/`)
- **Pydantic Settings** (`settings.py`): Type-safe configuration with environment variables
- **Multi-environment Support**: Development, testing, staging, production
- **Centralized Management**: All settings in one place

## 🔄 Data Source Management

The platform intelligently selects the best available data source based on your credentials:

### Source Priority Order
1. **FMP (Financial Modeling Prep)** - Highest quality, most comprehensive data
2. **WRDS (Wharton Research Data Services)** - Academic database with historical data
3. **Yahoo Finance** - Free, always available fallback

### Automatic Selection Logic
```python
# System automatically detects available sources
if FMP_API_KEY:
    use_fmp()        # Premium data
elif WRDS_USERNAME:
    use_wrds()      # Academic data
else:
    use_yahoo()     # Free data
```

### Data Source Comparison

| Feature | Yahoo Finance | FMP | WRDS |
|---------|---------------|-----|------|
| Cost | Free | Paid API | Academic Access |
| Historical Data | 2+ years | 10+ years | 50+ years |
| Fundamentals | Limited | Comprehensive | Extensive |
| Real-time | Limited | Good | Limited |
| Rate Limits | Moderate | Generous | None |

## 🔧 Configuration

The platform uses **Pydantic-based settings** with environment variable support:

### Environment Variables

```bash
# Application
ENVIRONMENT=development
APP_NAME="FinRL Trading"

# Alpaca API (Required for live trading)
APCA_API_KEY=your_alpaca_key
APCA_API_SECRET=your_alpaca_secret
APCA_BASE_URL=https://paper-api.alpaca.markets

# Data Sources (Optional, prioritized: FMP > WRDS > Yahoo)
FMP_API_KEY=your_fmp_api_key                    # Financial Modeling Prep
WRDS_USERNAME=your_wrds_username               # WRDS Database
WRDS_PASSWORD=your_wrds_password

# Risk Management
TRADING_MAX_ORDER_VALUE=100000
TRADING_MAX_PORTFOLIO_TURNOVER=0.5
STRATEGY_MAX_WEIGHT_PER_STOCK=0.1

# Data Management
DATA_CACHE_TTL_HOURS=24
DATA_MAX_CACHE_SIZE_MB=1000
```

### Configuration Structure

```python
from src.config.settings import get_config

config = get_config()
print(f"Environment: {config.environment}")
print(f"Database: {config.database.url}")
print(f"Risk Limits: {config.trading.max_order_value}")
```

## 📊 Performance Metrics

The backtesting engine provides comprehensive quantitative analysis:

### Return Metrics
- **Total Return**: Cumulative portfolio performance
- **Annualized Return**: Time-weighted annual performance
- **Alpha**: Excess return over benchmark

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted returns (Return ÷ Volatility)
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: Return ÷ Maximum Drawdown

### Tail Risk Measures
- **VaR (95%)**: Value at Risk at 95% confidence
- **CVaR (95%)**: Conditional Value at Risk
- **Skewness & Kurtosis**: Return distribution characteristics

### Benchmarking
- **Information Ratio**: Active return ÷ Tracking error
- **Beta**: Portfolio sensitivity to market
- **Tracking Error**: Standard deviation of active returns

## 🧪 Testing & Development

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run unit tests
pytest tests/

# Run with coverage report
pytest --cov=src --cov-report=html tests/

# Run specific test categories
pytest tests/test_strategies.py -v
pytest tests/test_trading.py -v

# Code quality checks
black src/                      # Format code
flake8 src/                     # Lint code
mypy src/                       # Type checking
```

### Test Coverage

The platform includes comprehensive tests for:
- Strategy implementations
- Backtesting engine
- Risk management
- Data processing pipeline
- API integrations

## 🛠️ Technology Stack

### Core Dependencies
- **Python 3.11+**: Modern Python with full type hints support
- **Pandas 2.0+**: High-performance data manipulation and analysis
- **NumPy 1.24+**: Advanced numerical computing
- **Scikit-learn 1.3+**: Comprehensive machine learning library
- **SciPy 1.11+**: Scientific computing and statistics

### Data Visualization & Web
- **Streamlit 1.28+**: Interactive web applications
- **Plotly 5.15+**: Advanced charting and visualization
- **Matplotlib 3.7+**: Publication-quality plotting
- **Seaborn 0.12+**: Statistical data visualization

### Financial APIs & Data
- **Alpaca-py 0.13+**: Live trading and market data API
- **WRDS Integration**: Academic financial database access
- **Requests 2.31+**: HTTP library for API calls

### Configuration & Validation
- **Pydantic v2.5+**: Data validation and settings management
- **Pydantic-settings 2.1+**: Environment-based configuration
- **Python-dotenv 1.0+**: Environment file support

### Database & Storage
- **SQLAlchemy 2.0+**: Modern ORM for database operations
- **SQLite**: Built-in database for data persistence

### Development & Testing
- **pytest 7.4+**: Comprehensive testing framework
- **pytest-cov 4.1+**: Code coverage reporting
- **Black 23.0+**: Code formatting
- **Flake8 6.1+**: Code linting
- **MyPy 1.7+**: Static type checking

### Optional: Advanced ML
- **TensorFlow 2.13+**: Deep learning framework
- **PyTorch 2.0+**: Research-grade deep learning
- **XGBoost 2.0+**: Gradient boosting framework
- **LightGBM 4.1+**: High-performance gradient boosting

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```
4. **Make your changes** with proper testing
5. **Run quality checks**
   ```bash
   black src/
   flake8 src/
   mypy src/
   pytest tests/
   ```
6. **Commit and push**
   ```bash
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

### Code Standards

- **Type Hints**: Use modern Python typing
- **Documentation**: Add docstrings to all public functions
- **Testing**: Write tests for new features
- **Style**: Follow PEP 8 with Black formatting

### Adding New Strategies

```python
from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult

class MyCustomStrategy(BaseStrategy):
    def generate_weights(self, data, **kwargs) -> StrategyResult:
        # Your strategy logic here
        pass
```

## 📋 Roadmap

### Current Features ✅
- ✅ Modular strategy framework
- ✅ ML-based stock selection
- ✅ Comprehensive backtesting
- ✅ Alpaca live trading integration
- ✅ Interactive web dashboard
- ✅ Docker containerization

### Planned Enhancements 🚧
- 🔄 Deep reinforcement learning strategies
- 🔄 Alternative data integration
- 🔄 Multi-asset support (crypto, futures)
- 🔄 Advanced risk management
- 🔄 Portfolio optimization algorithms
- 🔄 Real-time alerting system

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer

**⚠️ NOT FINANCIAL ADVICE**

This software is for **educational and research purposes only**. The algorithms, strategies, and tools provided:

- Are not guaranteed to be profitable
- May result in significant financial losses
- Should not be used for actual trading without thorough testing
- Do not constitute investment advice or recommendations

**Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.**

## 📚 References & Acknowledgments

### Academic Papers
- [Machine Learning for Stock Recommendation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3302088)
- [FinRL: Deep Reinforcement Learning Framework](https://arxiv.org/abs/2011.09607)
- [Portfolio Allocation with Deep Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)

### Open Source Projects
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [Alpaca-py](https://github.com/alpacahq/alpaca-py)
- [Streamlit](https://github.com/streamlit/streamlit)

### Data Sources
- [WRDS (Wharton Research Data Services)](https://wrds.wharton.upenn.edu/)
- [Alpaca Markets](https://alpaca.markets/)

---

**Built with ❤️ for the quantitative finance community**
