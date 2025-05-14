from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import logging
from logging.handlers import RotatingFileHandler

# 創建 Flask 應用
app = Flask(__name__)

# 配置日誌
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/wine_prediction.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Wine prediction app startup')

# 確保 models 目錄存在
if not os.path.exists('models'):
    os.mkdir('models')
    app.logger.warning('Models directory not found, created empty directory')

# 模型文件路徑
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'svc_model.pkl')


# 延遲加載模型函數
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        app.logger.info('Model loaded successfully')
        return model
    except FileNotFoundError:
        app.logger.error(f'Model file not found at {MODEL_PATH}')
        return None
    except Exception as e:
        app.logger.error(f'Error loading model: {str(e)}')
        return None


# 全局變數，用於存儲加載的模型
svc_model = None


@app.before_first_request
def initialize():
    """在第一個請求前初始化模型"""
    global svc_model
    svc_model = load_model()


@app.route('/')
def home():
    """首頁路由，顯示預測表單"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """接收預測請求並返回結果"""
    global svc_model

    # 如果模型未加載，嘗試加載
    if svc_model is None:
        svc_model = load_model()
        if svc_model is None:
            return jsonify({'error': 'Model not available'}), 503

    try:
        # 獲取並驗證請求數據
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # 提取並驗證特徵值
        try:
            alcohol = float(data.get('alcohol', 0))
            total_phenols = float(data.get('total_phenols', 0))
            color_intensity = float(data.get('color_intensity', 0))
            hue = float(data.get('hue', 0))
            proline = float(data.get('proline', 0))
        except (ValueError, TypeError) as e:
            app.logger.error(f'Input validation error: {str(e)}')
            return jsonify({'error': 'Invalid input values'}), 400

        # 記錄預測請求
        app.logger.info(f'Prediction request: alcohol={alcohol}, total_phenols={total_phenols}, '
                        f'color_intensity={color_intensity}, hue={hue}, proline={proline}')

        # 創建輸入數據並進行預測
        input_value = np.array([[alcohol, total_phenols, color_intensity, hue, proline]])
        pred_result = svc_model.predict(input_value)

        # 準備並返回結果
        result = {'prediction': int(pred_result[0])}
        app.logger.info(f'Prediction result: {result}')
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'Prediction error: {str(e)}')
        return jsonify({'error': 'Prediction failed'}), 500


@app.route('/health')
def health_check():
    """健康檢查端點，用於監控系統狀態"""
    return jsonify({'status': 'ok', 'model_loaded': svc_model is not None})


if __name__ == '__main__':
    # 在開發模式下啟動應用
    app.run(host='0.0.0.0', port=5000, debug=False)