from flask import Blueprint, request, render_template
import joblib
import numpy as np


# Tạo Blueprint
main = Blueprint('main', __name__)

# Tải mô hình khi khởi động ứng dụng
model = joblib.load('model/abalone_model.pkl')


@main.route('/')
def home():
    return render_template('index.html')  # Đảm bảo sử dụng đúng tên file template


@main.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[f"feature{i}"]) for i in range(1, 9)]
    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text=f'Kết quả dự đoán: {prediction[0]}')

