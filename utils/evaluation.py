import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

class ModelEvaluator:
    """模型评估类"""
    
    @staticmethod
    def evaluate_model(model, train_data, test_data, name=None, verbose=True):
        """
        评估单个模型的性能
        
        参数:
            model: 模型对象，需要有fit和predict方法
            train_data (Series): 训练数据
            test_data (Series): 测试数据
            name (str): 模型名称
            verbose (bool): 是否打印详细信息
            
        返回:
            dict: 评估结果
        """
        if name is None:
            name = model.__class__.__name__
            
        # 拟合模型
        start_time = time.time()
        model.fit(train_data)
        fit_time = time.time() - start_time
        
        # 预测
        pred = model.predict(len(test_data))
        
        # 计算评估指标
        mae = mean_absolute_error(test_data, pred)
        rmse = np.sqrt(mean_squared_error(test_data, pred))
        
        if verbose:
            print(f"{name}:")
            print(f"  MAE: {mae:.2f}°C")
            print(f"  RMSE: {rmse:.2f}°C")
            print(f"  训练时间: {fit_time:.2f}s")
            
        return {
            'name': name,
            'predictions': pred,
            'mae': mae,
            'rmse': rmse,
            'fit_time': fit_time
        }
    
    @staticmethod
    def compare_models(models, train_data, test_data, verbose=True):
        """
        比较多个模型的性能
        
        参数:
            models (dict): {模型名称: 模型对象}
            train_data (Series): 训练数据
            test_data (Series): 测试数据
            verbose (bool): 是否打印详细信息
            
        返回:
            DataFrame: 比较结果
        """
        results = []
        predictions = {}
        
        for name, model in models.items():
            result = ModelEvaluator.evaluate_model(
                model, train_data, test_data, name, verbose
            )
            results.append(result)
            predictions[name] = result['predictions']
            
        # 创建结果汇总表
        summary = pd.DataFrame(results)
        summary = summary[['name', 'mae', 'rmse', 'fit_time']]
        
        if verbose:
            print("\n模型性能对比:")
            print(summary)
            
        return summary, predictions