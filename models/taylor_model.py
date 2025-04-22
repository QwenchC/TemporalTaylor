import numpy as np
from scipy.integrate import cumulative_trapezoid
import time
import math

class TaylorSeriesModel:
    """
    基于泰勒多项式展开的时间序列预测模型
    
    核心思想：将时间序列在观测点t0处进行n阶泰勒展开，然后用多项式近似未来的序列值
    """
    
    def __init__(self, order=3, adaptive=False, max_order=5, threshold=0.1, max_horizon=30):
        """
        初始化泰勒多项式模型
        
        参数:
            order (int): 泰勒展开的阶数
            adaptive (bool): 是否启用自适应阶数选择
            max_order (int): 自适应模式下的最大阶数
            threshold (float): 自适应阶数选择的误差阈值
            max_horizon (int): 单次泰勒展开有效的最大预测步长
        """
        self.order = order
        self.adaptive = adaptive
        self.max_order = max_order
        self.threshold = threshold
        self.coefficients = None
        self.t0 = None
        self.fit_time = 0
        self.max_horizon = max_horizon  # 新增：限制有效预测范围
        
    def _compute_derivatives(self, series, t0, order):
        """
        计算时间序列在t0点的各阶导数
        
        参数:
            series (array): 时间序列数据
            t0 (int): 展开中心点索引
            order (int): 最高阶数
            
        返回:
            list: 各阶导数值
        """
        t = np.arange(len(series))
        # 使用插值函数使序列可微 - 添加平滑处理
        window = min(30, len(series) // 10)  # 平滑窗口
        smoothed = np.convolve(series, np.ones(window)/window, mode='valid')
        
        # 调整插值点，确保t0在有效范围内
        t_smooth = np.arange(len(smoothed))
        t0_smooth = min(max(0, t0 - window//2), len(smoothed)-1)
        
        f = lambda x: np.interp(x, t_smooth, smoothed)
        
        coeffs = [f(t0_smooth)]  # 0阶导数就是函数值
        
        # 实现数值导数计算 - 使用更稳定的方法
        def safe_numerical_derivative(func, x, h=1e-3):
            """更稳定的中心差分导数计算"""
            # 使用多点差分减小数值误差
            return (func(x + h) - func(x - h)) / (2 * h)
        
        # 计算高阶导数
        def safe_higher_derivative(func, x, n, h=1e-3):
            """稳定的高阶导数计算"""
            if n == 0:
                return func(x)
            elif n == 1:
                return safe_numerical_derivative(func, x, h)
            else:
                # 对于高阶导数，使用较大的步长以提高稳定性
                adaptive_h = h * (1 + 0.1 * (n-1))
                
                # 递归计算高阶导数
                def next_derivative(t):
                    return safe_higher_derivative(func, t, n-1, adaptive_h)
                
                return safe_numerical_derivative(next_derivative, x, adaptive_h)
        
        # 计算各阶导数（添加规范化以提高数值稳定性）
        scale_factor = np.max(np.abs(series)) or 1.0
        
        for k in range(1, order+1):
            try:
                # 计算导数并缩放
                coeff = safe_higher_derivative(f, t0_smooth, k)
                
                # 应用阶数衰减因子以降低高阶项的影响
                decay = np.exp(-0.5 * (k-1))
                coeff = coeff * decay
                
                # 检查并处理不稳定值
                if not np.isfinite(coeff) or abs(coeff) > 1e10:
                    print(f"警告: {k}阶导数值不稳定，设为0")
                    coeff = 0
                    
                coeffs.append(coeff)
            except Exception as e:
                print(f"计算{k}阶导数时出错: {e}")
                coeffs.append(0)  # 出错时使用0作为该阶导数
            
        return coeffs
    
    def _adaptive_order_selection(self, series, t0):
        """
        自适应选择泰勒展开的最佳阶数
        
        参数:
            series (array): 时间序列数据
            t0 (int): 展开中心点索引
            
        返回:
            int: 最佳阶数
        """
        if t0 >= len(series) - 1:
            return self.order  # 如果没有后续数据用于验证，返回默认阶数
            
        best_order = 1
        min_error = float('inf')
        
        # 从低阶开始尝试，找到最佳阶数
        for order in range(1, self.max_order+1):
            coeffs = self._compute_derivatives(series, t0, order)
            
            # 只预测一步，用于评估精度
            pred = self.predict_with_coeffs(coeffs, t0, 1)
            error = abs(pred[0] - series[t0+1])
            
            # 更新最佳阶数（优先选择低阶）
            if error < min_error:
                min_error = error
                best_order = order
                
            # 如果误差已经足够小，提前结束
            if error < self.threshold:
                break
                
        print(f"自适应选择的最佳阶数: {best_order}，误差: {min_error:.4f}")
        return best_order
    
    def fit(self, series):
        """
        拟合模型，计算泰勒展开系数
        
        参数:
            series (array): 时间序列数据
            
        返回:
            self
        """
        start_time = time.time()
        
        # 选择序列末尾作为展开中心点
        self.t0 = len(series) - 1
        
        # 如果启用自适应阶数，确定最佳阶数
        if self.adaptive:
            self.order = self._adaptive_order_selection(series, self.t0)
        
        # 计算各阶导数作为泰勒展开系数
        self.coefficients = self._compute_derivatives(series, self.t0, self.order)
        
        # 打印系数用于调试
        print(f"泰勒展开系数: {[f'{c:.4g}' for c in self.coefficients]}")
        
        self.fit_time = time.time() - start_time
        return self
    
    def predict_with_coeffs(self, coeffs, start, steps):
        """
        使用给定的泰勒系数生成预测序列
        
        参数:
            coeffs (list): 泰勒展开系数
            start (int): 起始点索引
            steps (int): 预测步数
            
        返回:
            array: 预测序列
        """
        prediction = np.zeros(steps, dtype=float)
        
        # 获取基准值（0阶系数）
        baseline = coeffs[0]
        
        # 对于每个预测步骤
        for i in range(steps):
            t_rel = i + 1  # 相对时间步长
            value = baseline  # 从基准值开始
            
            # 计算本步骤的预测值
            for k in range(1, len(coeffs)):
                # 限制远离展开点的预测
                if t_rel > self.max_horizon:
                    # 对于远离展开点的预测，使用线性外推
                    if k == 1:
                        term = coeffs[1] * self.max_horizon
                    else:
                        term = 0
                else:
                    # 计算泰勒项
                    try:
                        term = coeffs[k] * (t_rel**k) / math.factorial(k)
                        
                        # 处理数值溢出
                        if not np.isfinite(term):
                            print(f"警告: 泰勒项 k={k}, t={t_rel} 溢出，设置为0")
                            term = 0
                    except OverflowError:
                        print(f"警告: 计算 {t_rel}^{k} 溢出，设置为0")
                        term = 0
                
                value += term
            
            # 确保预测值在合理范围内（假设气温在-50到50°C之间）
            prediction[i] = max(min(value, 50), -50)
            
        return prediction
    
    def predict(self, steps):
        """
        预测未来steps个时间点的值
        
        参数:
            steps (int): 预测步数
            
        返回:
            array: 预测序列
        """
        if self.coefficients is None:
            raise ValueError("模型未拟合，请先调用fit方法")
        
        # 对于长期预测，使用分段预测
        if steps > self.max_horizon:
            print(f"预测步数({steps})超过单次泰勒展开的有效范围({self.max_horizon})，使用分段预测")
            return self._segment_predict(steps)
        
        return self.predict_with_coeffs(self.coefficients, self.t0, steps)
    
    def _segment_predict(self, steps):
        """
        进行分段预测
        
        参数:
            steps (int): 总预测步数
            
        返回:
            array: 预测序列
        """
        predictions = np.zeros(steps)
        remaining = steps
        current_pos = 0
        
        while remaining > 0:
            # 当前段预测步数
            seg_steps = min(remaining, self.max_horizon)
            
            # 计算当前段预测值
            segment_pred = self.predict_with_coeffs(self.coefficients, self.t0, seg_steps)
            predictions[current_pos:current_pos+seg_steps] = segment_pred
            
            # 更新位置和剩余步数
            current_pos += seg_steps
            remaining -= seg_steps
            
        return predictions
    
    def get_training_time(self):
        """
        返回模型训练时间
        
        返回:
            float: 训练时间（秒）
        """
        return self.fit_time
    
    def get_info(self):
        """
        返回模型信息
        
        返回:
            dict: 模型信息
        """
        return {
            "name": "Taylor Series Model",
            "order": self.order,
            "adaptive": self.adaptive,
            "coefficients": self.coefficients,
            "training_time": self.fit_time
        }

class MultiTaylorModel:
    """
    多段泰勒展开模型，通过多个固定长度窗口提高长期预测精度
    """
    
    def __init__(self, order=3, window_size=30):
        """
        初始化多段泰勒模型
        
        参数:
            order (int): 泰勒展开的阶数
            window_size (int): 每段泰勒展开的窗口大小
        """
        self.order = order
        self.window_size = window_size
        self.taylor_model = TaylorSeriesModel(order=order, max_horizon=window_size)
        self.fit_time = 0
        
    def fit(self, series):
        """
        拟合模型（保存训练数据）
        
        参数:
            series (array): 时间序列数据
            
        返回:
            self
        """
        start_time = time.time()
        self.series = series
        self.fit_time = time.time() - start_time
        return self
        
    def predict(self, steps):
        """
        预测未来steps个时间点的值，通过分段预测实现
        
        参数:
            steps (int): 预测步数
            
        返回:
            array: 预测序列
        """
        predictions = np.zeros(steps)
        segments = (steps + self.window_size - 1) // self.window_size
        
        series_copy = self.series.values if hasattr(self.series, 'values') else self.series.copy()
        
        for i in range(segments):
            # 每次预测一个窗口
            start_idx = i * self.window_size
            end_idx = min(start_idx + self.window_size, steps)
            segment_steps = end_idx - start_idx
            
            print(f"MultiTaylor: 预测分段 {i+1}/{segments}，步长 {segment_steps}")
            
            try:
                # 在当前数据末尾拟合泰勒模型
                self.taylor_model.fit(series_copy)
                
                # 预测下一个窗口
                segment_pred = self.taylor_model.predict(segment_steps)
                predictions[start_idx:end_idx] = segment_pred
                
                # 将预测结果添加到数据末尾，用于下一段预测
                series_copy = np.append(series_copy, segment_pred)
            except Exception as e:
                print(f"分段 {i+1} 预测出错: {e}")
                # 使用简单平均值作为预测
                if i == 0:
                    fallback_value = np.mean(series_copy[-30:])
                else:
                    fallback_value = predictions[start_idx-1]
                predictions[start_idx:end_idx] = fallback_value
                series_copy = np.append(series_copy, [fallback_value] * segment_steps)
            
        return predictions
        
    def get_training_time(self):
        """返回模型训练时间"""
        return self.fit_time