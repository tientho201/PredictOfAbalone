from flask import Flask, request, render_template
import joblib
import numpy as np
import xgboost as xgb

# Tạo Flask ứng dụng
app = Flask(__name__)

# Tải mô hình khi khởi động ứng dụng
model = joblib.load('model/abalone_model.pkl')

# Hàm chuyển đổi 'Sex' từ ký tự sang số
def convert_sex(sex):
    if sex == 'F':
        return 1
    elif sex == 'M':
        return 0
    elif sex == 'I':
        return 2
    else:
        return -1  # Giá trị không hợp lệ

@app.route('/')
def home():
    return render_template('index.html')  # Đảm bảo sử dụng đúng tên file template

@app.route('/predict', methods=['POST'])
def predict():
    # Chuyển đổi giá trị từ form
    sex = request.form.get("feature1")
    sex_value = convert_sex(sex)
    
    # Kiểm tra giá trị hợp lệ
    if sex_value == -1:
        return render_template('index.html', prediction_text="Giá trị giới tính không hợp lệ!")

    # Lấy các đặc trưng còn lại
    try:
        features = [sex_value] + [float(request.form[f"feature{i}"]) for i in range(2, 9)]
    except ValueError:
        return render_template('index.html', prediction_text="Vui lòng nhập số hợp lệ cho tất cả các đặc trưng!")

    # # Chuyển đổi thành DMatrix
    # final_features = xgb.DMatrix(np.array([features]))

    # Dự đoán
    prediction = model.predict(features)

    return render_template('index.html', prediction_text=f'Kết quả dự đoán: {prediction[0]:.4f}')  # Làm tròn kết quả dự đoán

if __name__ == '__main__':
    app.run(debug=True)
