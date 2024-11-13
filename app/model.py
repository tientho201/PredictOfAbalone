import pickle

def load_model():
    # Tải mô hình từ file
    with open('model/abalone_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_abalone_age(model, features):
    # Dự đoán dựa trên các đặc trưng đầu vào
    prediction = model.predict([features])
    return int(prediction[0])  # Trả về giá trị nguyên
