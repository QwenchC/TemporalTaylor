import numpy as np
import pandas as pd
import time
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import warnings

class ARIMAModel:
    """ARIMA模型封装类"""
    
    def __init__(self, order=(2, 1, 2)):
        """
        初始化ARIMA模型
        
        参数:
            order (tuple): ARIMA模型阶数(p,d,q)
        """
        self.order = order
        self.model = None
        self.fit_time = 0
        
    def fit(self, series):
        """
        拟合ARIMA模型
        
        参数:
            series (Series): 时间序列数据
            
        返回:
            self
        """
        start_time = time.time()
        
        # 忽略警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = ARIMA(series, order=self.order).fit()
            
        self.fit_time = time.time() - start_time
        return self
        
    def predict(self, steps):
        """
        预测未来steps个时间点的值
        
        参数:
            steps (int): 预测步数
            
        返回:
            array: 预测序列
        """
        if self.model is None:
            raise ValueError("模型未拟合，请先调用fit方法")
            
        forecast = self.model.forecast(steps)
        return forecast.values
        
    def get_training_time(self):
        """返回模型训练时间"""
        return self.fit_time
        
    def get_info(self):
        """返回模型信息"""
        return {
            "name": "ARIMA Model",
            "order": self.order,
            "training_time": self.fit_time
        }


class LSTMModel:
    """LSTM模型封装类"""
    
    def __init__(self, window_size=30, units=50, epochs=20):
        """
        初始化LSTM模型
        
        参数:
            window_size (int): 输入窗口大小
            units (int): LSTM单元数量
            epochs (int): 训练轮数
        """
        self.window_size = window_size
        self.units = units
        self.epochs = epochs
        self.model = None
        self.fit_time = 0
        
    def _create_dataset(self, data):
        """
        创建滑动窗口数据集
        
        参数:
            data (array): 时间序列数据
            
        返回:
            tuple: (X, y)
        """
        X, y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size])
        return np.array(X), np.array(y)
        
    def _build_model(self):
        """
        构建LSTM模型
        
        返回:
            Model: Keras模型
        """
        model = Sequential([
            LSTM(self.units, input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def fit(self, series):
        """
        拟合LSTM模型
        
        参数:
            series (Series): 时间序列数据
            
        返回:
            self
        """
        start_time = time.time()
        
        # 准备数据
        data = series.values
        X, y = self._create_dataset(data)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # 构建并训练模型
        self.model = self._build_model()
        
        # 修改: 显示训练进度 (verbose=1)
        print(f"开始训练LSTM模型 (epochs={self.epochs})...")
        self.model.fit(X, y, epochs=self.epochs, verbose=1)
        
        # 保存最后一个窗口用于预测
        self.last_window = data[-self.window_size:]
        
        self.fit_time = time.time() - start_time
        return self
        
    def predict(self, steps):
        """
        预测未来steps个时间点的值
        
        参数:
            steps (int): 预测步数
            
        返回:
            array: 预测序列
        """
        if self.model is None:
            raise ValueError("模型未拟合，请先调用fit方法")
            
        # 初始化预测窗口
        curr_window = self.last_window.copy()
        predictions = []
        
        for _ in range(steps):
            # 重塑窗口形状用于LSTM输入
            x = curr_window.reshape(1, self.window_size, 1)
            # 预测下一个值
            next_pred = self.model.predict(x, verbose=0)[0][0]
            # 添加到预测结果
            predictions.append(next_pred)
            # 更新窗口
            curr_window = np.append(curr_window[1:], next_pred)
            
        return np.array(predictions)
        
    def get_training_time(self):
        """返回模型训练时间"""
        return self.fit_time
        
    def get_info(self):
        """返回模型信息"""
        return {
            "name": "LSTM Model",
            "window_size": self.window_size,
            "units": self.units,
            "epochs": self.epochs,
            "training_time": self.fit_time
        }