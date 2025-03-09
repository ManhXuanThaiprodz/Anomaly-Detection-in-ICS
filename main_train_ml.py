import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from data_loader import load_train_data, load_test_data
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ========================== 1️⃣ CHUYỂN DỮ LIỆU TIME-SERIES → TABULAR ==========================

def extract_features(X, window_size=10):
    """
    Chuyển đổi dữ liệu chuỗi thời gian thành dạng bảng bằng cách tính các đặc trưng thống kê.
    """
    feature_list = []
    
    for i in range(window_size, X.shape[0]):  
        window = X[i-window_size:i]  # Lấy một cửa sổ dữ liệu
        
        # Tính toán các đặc trưng thống kê
        feature_vector = [
            np.mean(window, axis=0),  # Giá trị trung bình
            np.std(window, axis=0),   # Độ lệch chuẩn
            np.min(window, axis=0),   # Min
            np.max(window, axis=0),   # Max
            np.median(window, axis=0) # Trung vị
        ]
        
        # Ghép tất cả đặc trưng thành một vector
        feature_list.append(np.concatenate(feature_vector))
    
    return np.array(feature_list)

# ========================== 2️⃣ HÀM CHỌN MÔ HÌNH ML ==========================

def get_ml_model(model_type):
    """
    Trả về thuật toán ML Regression phù hợp.
    """
    if model_type == "SVM":
        return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    elif model_type == "RF":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "AdaBoost":
        return AdaBoostRegressor(n_estimators=50, random_state=42)
    elif model_type == "XGBoost":
        return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose from: SVM, RF, AdaBoost, XGBoost")

# ========================== 3️⃣ HÀM HUẤN LUYỆN MÔ HÌNH ML ==========================

def train_ml_model(model_type, X_train, X_test, Y_test):
    """
    Huấn luyện mô hình ML Regression và tính toán lỗi tái tạo.
    - Sử dụng X_train để huấn luyện mô hình.
    - Mô hình dự đoán chính X_train để học cách tái tạo dữ liệu bình thường.
    """
    model = get_ml_model(model_type)

    # Huấn luyện mô hình (không cần Y_train)
    print(f"\n🚀 Training {model_type} model...")
    model.fit(X_train, X_train.mean(axis=1))  # Dùng chính X_train để học cách tái tạo

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Tính toán lỗi tái tạo (Reconstruction Error)
    reconstruction_error = ((X_test.mean(axis=1) - y_pred) ** 2)

    # Đánh giá mô hình
    mse = mean_squared_error(X_test.mean(axis=1), y_pred)
    r2 = r2_score(X_test.mean(axis=1), y_pred)

    print(f"\n📊 {model_type} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    return model, reconstruction_error

# ========================== 4️⃣ MAIN SCRIPT ==========================

if __name__ == "__main__":
    # Nhận tham số từ dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml", type=str, default="RF", help="Choose ML model: SVM, RF, AdaBoost, XGBoost")
    args = parser.parse_args()

    # Load dữ liệu từ repo GitHub
    dataset_name = "BATADAL"
    X_train, sensor_cols = load_train_data(dataset_name)  # X_train không có nhãn Y_train
    X_test, Y_test, _ = load_test_data(dataset_name)  # Tập kiểm tra có nhãn

    # Chuyển đổi dữ liệu chuỗi thời gian thành dạng bảng
    window_size = 10
    X_train_features = extract_features(X_train, window_size)
    X_test_features = extract_features(X_test, window_size)

    print("\n📌 Dữ liệu sau khi chuyển đổi:")
    print("🔹 X_train_features shape:", X_train_features.shape)
    print("🔹 X_test_features shape:", X_test_features.shape)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_features = scaler.fit_transform(X_train_features)
    X_test_features = scaler.transform(X_test_features)

    # Nếu có chọn ML, thì huấn luyện ML
    if args.ml:
        model, reconstruction_error = train_ml_model(args.ml, X_train_features, X_test_features, Y_test)

        # Xác định ngưỡng bất thường (95th percentile)
        threshold = np.percentile(reconstruction_error, 95)
        anomalies = (reconstruction_error > threshold).astype(int)

        # ========================== 5️⃣ ĐÁNH GIÁ KẾT QUẢ ==========================

        print("\n📌 Classification Report:")
        print(classification_report(Y_test, anomalies))

        print(f"\n✅ Số điểm bất thường phát hiện được: {sum(anomalies)} / {len(Y_test)}")
        print(f"📊 Ngưỡng bất thường: {threshold:.4f}")

        # ========================== 6️⃣ TRỰC QUAN HÓA ==========================

        plt.figure(figsize=(8, 5))
        plt.hist(reconstruction_error, bins=50, color='blue', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='dashed', label='Threshold')
        plt.title("Lỗi tái tạo (Reconstruction Error)")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
