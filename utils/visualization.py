import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import matplotlib
import os

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建结果目录
RESULT_DIR = "./result"
os.makedirs(RESULT_DIR, exist_ok=True)

class DataVisualizer:
    """数据可视化类"""
    
    @staticmethod
    def plot_time_series(series, title="Time Series Data", figsize=(12, 6), save_path=None):
        """
        绘制时间序列数据
        
        参数:
            series (Series): 时间序列数据
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        plt.figure(figsize=figsize)
        plt.plot(series.index, series.values)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_train_test_split(train, test, title="Train-Test Split", figsize=(12, 6), save_path=None):
        """
        可视化训练集和测试集
        
        参数:
            train (Series): 训练数据
            test (Series): 测试数据
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        plt.figure(figsize=figsize)
        plt.plot(train.index, train.values, label='Train')
        plt.plot(test.index, test.values, label='Test')
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_predictions(test_data, predictions, title="Model Predictions", figsize=(12, 6), save_path=None):
        """
        绘制预测结果对比图
        
        参数:
            test_data (Series): 真实测试数据
            predictions (dict): {模型名称: 预测值}
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        plt.figure(figsize=figsize)
        
        # 绘制真实值
        plt.plot(test_data.index, test_data.values, 'k-', label='True', linewidth=2)
        
        # 绘制各模型预测值
        colors = ['b', 'r', 'g', 'm', 'c']
        for i, (name, pred) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            plt.plot(test_data.index, pred, color, label=name, alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_error_metrics(results, title="Model Comparison", figsize=(10, 6), save_path=None):
        """
        绘制模型评估指标对比图
        
        参数:
            results: 评估结果(DataFrame或字典列表)
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        # 检查results是DataFrame还是字典列表
        if isinstance(results, pd.DataFrame):
            # 使用DataFrame的值
            names = results['name'].tolist()
            mae_values = results['mae'].tolist()
            rmse_values = results['rmse'].tolist()
        else:
            # 假设是字典列表
            names = [r.get('name', '') for r in results]
            mae_values = [r.get('mae', 0) for r in results]
            rmse_values = [r.get('rmse', 0) for r in results]
        
        # 设置图表
        x = np.arange(len(names))
        width = 0.35
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=figsize)
        bars1 = ax.bar(x - width/2, mae_values, width, label='MAE')
        bars2 = ax.bar(x + width/2, rmse_values, width, label='RMSE')
        
        # 添加标签和标题
        ax.set_ylabel('Error (°C)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        
        # 添加数值标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        add_labels(bars1)
        add_labels(bars2)
        
        fig.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_taylor_coefficients(coeffs, order=None, title="Taylor Series Coefficients", figsize=(10, 6), save_path=None):
        """
        可视化泰勒展开系数
        
        参数:
            coeffs (list): 泰勒展开系数
            order (int): 展开阶数
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        if order is None:
            order = len(coeffs) - 1
            
        plt.figure(figsize=figsize)
        
        labels = ['常数项', '一阶导数', '二阶导数', '三阶导数', '四阶导数', '五阶导数']
        labels = labels[:len(coeffs)]
        
        plt.bar(labels, coeffs)
        plt.title(title)
        plt.ylabel("系数值")
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(coeffs):
            plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
            
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_training_times(results, title="Model Training Times", figsize=(10, 6), save_path=None):
        """
        绘制模型训练时间对比图
        
        参数:
            results: 评估结果(DataFrame或字典列表)
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        plt.figure(figsize=figsize)
        
        # 检查results是DataFrame还是字典列表
        if isinstance(results, pd.DataFrame):
            # 使用DataFrame的值
            names = results['name'].tolist()
            times = results['fit_time'].tolist()
        else:
            # 假设是字典列表
            names = [r.get('name', '') for r in results]
            times = [r.get('training_time', 0) for r in results]
        
        plt.bar(names, times)
        plt.title(title)
        plt.ylabel("Time (seconds)")
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, v in enumerate(times):
            plt.text(i, v, f"{v:.2f}s", ha='center', va='bottom')
            
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()
    
    @staticmethod
    def plot_error_over_horizons(test_data, predictions, title="Prediction Error Over Horizons", figsize=(12, 6), save_path=None):
        """
        绘制不同预测时长的误差变化
        
        参数:
            test_data (Series): 真实测试数据
            predictions (dict): {模型名称: 预测值}
            title (str): 图表标题
            figsize (tuple): 图表大小
            save_path (str): 保存文件路径，如果为None则自动生成
        """
        horizons = [1, 7, 30, 90, 180]  # 1天，1周，1月，3月，6月
        max_horizon = min(len(test_data), 180)
        
        mae_by_horizon = {}
        
        for name, pred in predictions.items():
            mae_by_horizon[name] = []
            
            for h in horizons:
                if h > max_horizon:
                    break
                # 计算不同预测长度的MAE
                mae = mean_absolute_error(test_data[:h].values, pred[:h])
                mae_by_horizon[name].append(mae)
                
        # 绘制图表
        plt.figure(figsize=figsize)
        
        for name, errors in mae_by_horizon.items():
            plt.plot(horizons[:len(errors)], errors, 'o-', label=name)
            
        plt.title(title)
        plt.xlabel("Prediction Horizon (days)")
        plt.ylabel("MAE (°C)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 自动生成保存路径
        if save_path is None:
            save_path = os.path.join(RESULT_DIR, f"{title.replace(' ', '_')}.png")
        
        # 保存图表
        plt.savefig(save_path)
        print(f"图表已保存至: {save_path}")
        
        try:
            plt.show()
        except Exception as e:
            print(f"显示图表时出错: {e}")
        
        plt.close()