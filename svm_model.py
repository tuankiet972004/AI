import os
import joblib
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục chứa dữ liệu hình ảnh
data_dir = 'data_char'  # Thay đổi đường dẫn này

# Danh sách để lưu các đặc trưng và nhãn
features = []
labels = []

# Đọc hình ảnh từ các thư mục
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        label = int(folder) if int(folder) < 10 else chr(ord('A') + int(folder) - 10)
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (32, 32))
            features.append(img_resized.flatten())
            labels.append(label)

# Chuyển đổi thành mảng NumPy
X = np.array(features)
y = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình SVM
model = svm.SVC(kernel='linear')

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Lưu mô hình vào file
joblib.dump(model, 'model/svm_model.pkl')

# Tải mô hình từ file
loaded_model = joblib.load('model/svm_model.pkl')

# Bây giờ bạn có thể sử dụng loaded_model để dự đoán
predicted_label = loaded_model.predict(X_test)
print(f"Dự đoán cho tập kiểm tra: {predicted_label}")

# Hàm để dự đoán ký tự từ hình ảnh
def predict_character(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (32, 32))  # Thay đổi kích thước về 32x32
    img_flattened = img_resized.flatten().reshape(1, -1)  # Chuyển đổi thành mảng 2D
    prediction = model.predict(img_flattened)
    return prediction[0]  # Trả về nhãn dự đoán

# Đường dẫn đến hình ảnh bạn muốn kiểm tra
test_image_path = 'data_char/03/1.jpg'  # Thay đổi đường dẫn này
predicted_label = predict_character(test_image_path, loaded_model)
print(f"Dự đoán cho hình ảnh: {predicted_label}")
