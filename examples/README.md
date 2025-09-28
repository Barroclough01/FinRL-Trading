# FinRL Trading 示例

本文件夹包含了完整的FinRL Trading使用示例，展示从数据获取到交易执行的完整工作流。

## 📁 文件说明

### [FinRL_Full_Workflow.ipynb](FinRL_Full_Workflow.ipynb)
**完整的交互式教程** - 推荐从这里开始

展示完整的量化交易工作流：
- ✅ 数据获取（S&P 500成分股、基本面数据、历史股价）
- ✅ 机器学习选股策略
- ✅ 策略回测（对比VOO和QQQ基准）
- ✅ Alpaca Paper Trading执行

**使用方法：**
```bash
# 安装依赖
pip install -r ../requirements.txt

# 配置环境变量
cp ../.env.example ../.env
# 编辑.env文件，填入API密钥

# 启动Jupyter
jupyter notebook FinRL_Full_Workflow.ipynb
```

### 即将添加的其他示例：
- `simple_backtest.py` - 简单回测示例
- `data_fetching_demo.py` - 数据获取演示
- `ml_strategy_example.py` - ML策略示例
- `live_trading_example.py` - 实盘交易示例

## 🚀 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/your-repo/FinRL-Trading.git
cd FinRL-Trading

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥
```bash
# 复制环境配置模板
cp .env.example .env

# 编辑配置文件
nano .env
```

必需的API配置：
- **Alpaca API**: 用于交易（必需）
  - 获取密钥: https://alpaca.markets/
- **数据源API**: 用于高质量数据（可选）
  - FMP: https://financialmodelingprep.com/
  - WRDS: https://wrds-www.wharton.upenn.edu/

### 3. 运行示例
```bash
# 启动Jupyter Notebook
jupyter notebook examples/FinRL_Full_Workflow.ipynb

# 或者运行Python脚本
python examples/simple_example.py
```

## 📊 示例特色

### 🔄 完整工作流
1. **数据获取层**: 多数据源自动选择（FMP > WRDS > Yahoo）
2. **策略层**: 机器学习选股策略
3. **回测层**: 专业的策略评估和基准对比
4. **交易层**: Alpaca Paper Trading集成

### 🎯 核心功能
- **智能数据源选择**: 自动选择最佳可用数据源
- **机器学习选股**: 基于特征的股票评分和选择
- **专业回测**: 完整的风险指标计算和基准对比
- **模拟交易**: Alpaca Paper Trading安全测试

### 📈 支持的指标
- 总收益率 / 年化收益率
- 夏普比率 / 索提诺比率
- 最大回撤 / 恢复时间
- VaR / CVaR 风险度量
- 胜率 / 盈亏比

## ⚙️ 配置说明

### 数据源优先级
```
FMP (Financial Modeling Prep)     ⭐⭐⭐⭐⭐
    └─ 高质量付费数据，推荐用于生产环境

WRDS (Wharton Research)          ⭐⭐⭐⭐☆
    └─ 学术数据库，适合研究用途

Yahoo Finance                    ⭐⭐⭐☆☆
    └─ 免费数据，速率限制较多
```

### 环境变量
```bash
# 必需
APCA_API_KEY=your_alpaca_key
APCA_API_SECRET=your_alpaca_secret

# 可选 - 提高数据质量
FMP_API_KEY=your_fmp_key
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

## 🛠️ 故障排除

### 常见问题

1. **数据获取失败**
   ```bash
   # 检查网络连接
   ping google.com

   # 验证API配置
   python -c "from src.config.settings import get_config; print(get_config().alpaca.api_key)"
   ```

2. **Alpaca连接问题**
   ```bash
   # 验证API密钥
   # 检查账户状态
   # 确认使用paper trading
   ```

3. **回测结果异常**
   ```bash
   # 检查数据质量
   # 验证权重计算
   # 审查交易成本设置
   ```

### 性能优化
- 使用缓存减少API调用
- 批量处理数据请求
- 优化特征选择
- 定期重新训练模型

## 📚 学习路径

1. **新手入门**: 从 `complete_trading_workflow.ipynb` 开始
2. **进阶学习**: 研究ML策略和风险管理
3. **生产部署**: 配置监控和自动化
4. **策略开发**: 创建自定义策略和指标

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本项目采用 MIT 许可证。
