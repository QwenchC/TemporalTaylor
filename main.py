import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from datetime import datetime

import matplotlib
matplotlib.use('TkAgg')  # 尝试使用TkAgg后端

# 导入自定义模块
from utils.data_loader import DataLoader
from utils.evaluation import ModelEvaluator
from utils.visualization import DataVisualizer
from models.taylor_model import TaylorSeriesModel, MultiTaylorModel
from models.baseline_models import ARIMAModel, LSTMModel

# 忽略警告
warnings.filterwarnings('ignore')

def main():
    """主程序入口"""
    print("=" * 50)
    print("基于泰勒展开的时间序列预测项目")
    print("=" * 50)
    
    # 1. 加载数据
    data_loader = DataLoader()
    df = data_loader.download_denver_temperature()
    
    # 2. 预处理数据
    series = data_loader.preprocess(df)
    
    # 3. 可视化原始数据
    print("\n原始时间序列数据:")
    DataVisualizer.plot_time_series(series, title="丹佛日均气温 (1995-2020)")
    
    # 4. 划分训练集和测试集
    train, test = data_loader.train_test_split(series)
    print(f"训练集大小: {len(train)}, 测试集大小: {len(test)}")
    DataVisualizer.plot_train_test_split(train, test, title="训练集与测试集划分")
    
    # 5. 定义模型
    print("\n定义模型...")
    models = {
        "ARIMA": ARIMAModel(order=(2, 1, 2)),
        "LSTM": LSTMModel(window_size=30, units=25, epochs=5),  # 降低units和epochs
        "Taylor3": TaylorSeriesModel(order=3),
        "Taylor5": TaylorSeriesModel(order=5),
        "MultiTaylor": MultiTaylorModel(order=3, window_size=30)
    }
    
    # 6. 评估模型
    print("\n开始模型评估:")
    
    # 添加更详细的进度打印
    model_count = len(models)
    for i, (model_name, model) in enumerate(models.items()):
        print(f"\n正在评估模型 [{i+1}/{model_count}]: {model_name}")
        if model_name == "ARIMA":
            print("ARIMA模型训练中，这可能需要一些时间...")
        elif model_name == "LSTM":
            print("LSTM模型准备训练，将显示训练进度...")
        else:
            print(f"{model_name}模型训练中...")
            
    # 原有的模型评估代码
    summary, predictions = ModelEvaluator.compare_models(models, train, test)
    
    # 7. 可视化预测结果
    print("\n正在生成可视化图表(1/6): 短期预测对比...")
    # 前90天预测
    short_test = test[:90]
    short_preds = {name: pred[:90] for name, pred in predictions.items()}
    DataVisualizer.plot_predictions(
        short_test, 
        short_preds, 
        title="3个月短期预测对比 (2016年1-3月)"
    )
    
    # 其他可视化代码也添加类似的进度提示
    print("\n正在生成可视化图表(2/6): 长期预测对比...")
    # 全部预测
    DataVisualizer.plot_predictions(
        test, 
        predictions, 
        title="长期预测对比 (2016-2020)"
    )
    
    print("\n正在生成可视化图表(3/6): 模型误差对比...")
    # 8. 可视化评估指标
    DataVisualizer.plot_error_metrics(summary, title="模型误差对比")
    
    print("\n正在生成可视化图表(4/6): 模型训练时间对比...")
    DataVisualizer.plot_training_times(summary, title="模型训练时间对比")
    
    print("\n正在生成可视化图表(5/6): 泰勒展开系数分析...")
    # 9. 泰勒模型系数分析
    taylor_model = models["Taylor3"]
    taylor_model.fit(train)
    coeffs = taylor_model.coefficients
    
    print("\n泰勒展开系数分析:")
    print(f"常数项 (f(t0)): {coeffs[0]:.4f}°C - 展开点的温度值")
    print(f"一阶导数 (f'(t0)): {coeffs[1]:.4f}°C/天 - 温度变化率")
    print(f"二阶导数 (f''(t0)): {coeffs[2]:.4f}°C/天² - 温度变化加速度")
    print(f"三阶导数 (f'''(t0)): {coeffs[3]:.4f}°C/天³ - 温度变化加加速度")
    
    DataVisualizer.plot_taylor_coefficients(
        coeffs, 
        title="3阶泰勒展开系数"
    )
    
    print("\n正在生成可视化图表(6/6): 不同预测长度的误差变化...")
    # 10. 分析预测误差随时间的变化
    DataVisualizer.plot_error_over_horizons(
        test, 
        predictions, 
        title="不同预测长度的误差变化"
    )
    
    # 11. 自适应阶数泰勒模型示例
    print("\n自适应阶数泰勒模型:")
    adaptive_taylor = TaylorSeriesModel(adaptive=True, max_order=5, threshold=0.2)
    adaptive_result = ModelEvaluator.evaluate_model(
        adaptive_taylor, train, test[:90], "自适应泰勒模型"
    )
    
    print("\n完成！项目执行结束。")
    
    # 确保所有图形都能显示
    plt.show(block=True)
    
    # 如果上面的方法不起作用，尝试下面的替代方案
    import os
    
    print("\n如果没有看到图表窗口，可尝试在 ./plots 目录查看保存的图表文件")
    plots_dir = "./plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 生成一个测试图表确保图形功能正常
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3, 4, 5], [2, 3, 5, 7, 11], 'ro-')
    plt.title("测试图表 - 请确认您能看到此图")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.savefig(os.path.join(plots_dir, "test_plot.png"))
    plt.show()

if __name__ == "__main__":
    main()