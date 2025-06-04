import cv2
import joblib
import numpy as np

# Tải mô hình từ file
model = joblib.load('model/svm_model.pkl')


def predict_character(img, model):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (32, 32))  # Đảm bảo kích thước chuẩn cho mô hình
    img_flattened = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flattened)
    return str(prediction[0])

def extract_and_predict_numbers(image_path, model):
    img = cv2.imread(image_path)

    fixed_size = (500, 160)
    img = cv2.resize(img, fixed_size)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)  # Điều chỉnh ngưỡng

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp contours dựa trên vị trí x
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    predicted_numbers = []
    img_contours = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Kiểm tra kích thước contour
        area = w * h
        if w > 10 and h > 20 and area > 1500 and area < 10000:  # Thêm điều kiện diện tích
            # In ra kích thước của contour
            print(f"Kích thước contour: width = {w}, height = "
                  f"{h}, area = {area}")

            padding = 8
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = x + w + padding
            y_end = y + h + padding

            roi = img[y_start:y_end, x_start:x_end]
            predicted_character = predict_character(roi, model)
            print(f"Dự đoán ký tự: {predicted_character}")
            predicted_numbers.append(predicted_character)

            cv2.rectangle(img_contours, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imshow('Contours', img_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ''.join(predicted_numbers)

# Đường dẫn đến hình ảnh chứa dãy chữ số
image_path = 'cropped_license_plate.jpg'
result = extract_and_predict_numbers(image_path, model)
print(f"Dự đoán: {result}")
