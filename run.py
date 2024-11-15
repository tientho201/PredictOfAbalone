from flask import Flask, request, render_template
import joblib
import numpy as np
import xgboost as xgb

# Tạo Flask ứng dụng
app = Flask(__name__)

# Tải mô hình khi khởi động ứng dụng
model_cat = joblib.load('model/abalone_model_cat.pkl')
model_xgb = joblib.load('model/abalone_model_xgb.pkl')

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
    model_choice = request.form.get("model")
    # Chuyển đổi giá trị từ form
    sex = request.form.get("feature1")
    sex_value = convert_sex(sex)
    
    # Kiểm tra giá trị hợp lệ
    if sex_value == -1:
        return render_template('index.html', prediction_text="Giá trị giới tính không hợp lệ!")
    # Thu thập các đặc trưng đầu vào
    length = float(request.form["feature2"])
    diameter = float(request.form["feature3"])
    height = float(request.form["feature4"])
    whole_weight = float(request.form["feature5"])
    shucked_weight = float(request.form["feature6"])
    viscera_weight = float(request.form["feature7"])
    shell_weight = float(request.form["feature8"])

    # Áp dụng feature engineering
    diameter_to_height_ratio = diameter / height if height != 0 else 0
    combined_whole_weight = whole_weight + shucked_weight + viscera_weight
    diameter_length_product = diameter * length
    shell_volume = (4/3) * 3.14 * (diameter / 2)**2 * height
    shell_surface_area = 4 * 3.14 * (diameter / 2)**2
    shell_density = shell_weight / shell_volume if shell_volume != 0 else 0
    shell_thickness = height - diameter
    shell_shape_index = shell_surface_area / shell_volume if shell_volume != 0 else 0
    length_to_height_ratio = length / height if height != 0 else 0
    # Lấy các đặc trưng còn lại
    try:
         # Tạo mảng đặc trưng hoàn chỉnh
        features =   [sex_value , 
            length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight,
        diameter_to_height_ratio, combined_whole_weight, diameter_length_product,
        shell_volume, shell_surface_area, shell_density, shell_thickness, shell_shape_index,
        length_to_height_ratio
    
    ]

    except ValueError:
        return render_template('index.html', prediction_text="Vui lòng nhập số hợp lệ cho tất cả các đặc trưng!")

    # # Dự đoán
    # prediction = model.predict(features)

    # return render_template('index.html', prediction_text=f'Kết quả dự đoán: {prediction[0]:.4f}')  # Làm tròn kết quả dự đoán
    # Chọn mô hình
    if model_choice == "xgb":
        prediction = model_xgb.predict([features])
    elif model_choice == "catboost":
        prediction = model_cat.predict([features])

    return render_template("index.html", prediction_text=f"Kết quả dự đoán: {prediction[0]:.4f}")
if __name__ == '__main__':
    app.run(debug=True)
