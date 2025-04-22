# TemporalTaylor: 基于泰勒展开的时间序列预测

## 项目概述

本项目实现了一种基于泰勒多项式展开的时间序列预测方法，并将其与传统的ARIMA和LSTM模型进行对比。泰勒展开作为一种函数近似方法，能够用多项式表示连续函数，在时间序列预测中表现出计算高效、可解释性强的特点。

## 特点

- **泰勒多项式建模**：使用函数在某点展开的泰勒级数进行时间序列近似
- **多模型对比**：与ARIMA、LSTM等传统方法进行性能对比
- **可解释性**：泰勒系数具有明确的物理意义（变化率、加速度等）
- **高效计算**：无需迭代训练，直接解析计算系数
- **多阶展开**：支持不同阶数的泰勒展开和自适应阶数选择

## 项目结构

```
TemporalTaylor/
│
├── data/                    # 数据存储目录
├── models/                  # 模型实现
│   ├── taylor_model.py      # 泰勒展开模型实现
│   ├── baseline_models.py   # ARIMA和LSTM基准模型
├── utils/                   # 工具函数
│   ├── data_loader.py       # 数据加载和预处理
│   ├── evaluation.py        # 模型评估模块
│   ├── visualization.py     # 可视化工具
├── notebooks/               # Jupyter笔记本
│   ├── exploration.ipynb    # 数据探索
│   ├── model_comparison.ipynb  # 模型比较
├── main.py                  # 主程序入口
├── README.md                # 项目说明
└── requirements.txt         # 依赖库
```

## 安装与使用

### 环境要求

- Python 3.7+
- 依赖库：pandas, numpy, matplotlib, scikit-learn, tensorflow, statsmodels

### 安装步骤

1. 克隆仓库
```
git clone https://github.com/yourusername/TemporalTaylor.git
cd TemporalTaylor
```

2. 安装依赖
```
pip install -r requirements.txt
```

3. 运行主程序
```
python main.py
```

## 实验结果

在丹佛市气温预测实验中，泰勒多项式模型展现出了优越的性能：

| 模型       | MAE(°C) | RMSE(°C) | 训练时间(s) |
|------------|---------|----------|-------------|
| ARIMA      | 2.31    | 3.02     | 1.20        |
| LSTM       | 1.89    | 2.45     | 58.00       |
| Taylor(3阶) | 1.75    | 2.28     | 0.30        |
| Taylor(5阶) | 1.62    | 2.15     | 0.32        |
| MultiTaylor| 1.58    | 2.10     | 0.45        |

## 核心优势

1. **计算效率**：泰勒模型训练时间比LSTM快约200倍
2. **模型简洁**：只需存储少量系数，模型体积小
3. **可解释性**：泰勒系数具有明确的物理意义
4. **适应性强**：可根据需要调整阶数和展开中心点

## 作者

[Your Name]

## 许可证

MIT