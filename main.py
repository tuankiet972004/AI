import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import datetime
from ultralytics import YOLO
import cv2
import joblib
import numpy as np

# Biến toàn cục để lưu đường dẫn ảnh đã chọn
selected_image_path = None

# Tải mô hình từ file
model = joblib.load('model/svm_model.pkl')

def choose_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        selected_image_path = file_path  # Lưu đường dẫn của ảnh đã chọn
        img = Image.open(file_path)
        img = img.resize((400, 300))  # Đảm bảo kích thước ảnh là 400x300 pixel
        img_tk = ImageTk.PhotoImage(img)

        image_label.config(image=img_tk, width=400, height=300)
        image_label.image = img_tk  # Giữ tham chiếu để không bị xóa bởi Garbage Collector

        # Cập nhật ngày giờ thêm ảnh
        date_time_label.config(text=f"Ngày giờ thêm: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

def predict_character(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (32, 32))  # Đảm bảo kích thước chuẩn cho mô hình
    img_flattened = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flattened)
    return str(prediction[0])

def extract_and_predict_numbers(image_path):
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
        if w > 10 and w < 60 and h > 60 and h < 150 and area > 1500 and area < 10000:  # Thêm điều kiện diện tích
            padding = 8
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = x + w + padding
            y_end = y + h + padding

            roi = img[y_start:y_end, x_start:x_end]
            predicted_character = predict_character(roi)
            predicted_numbers.append(predicted_character)

            cv2.rectangle(img_contours, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # Chuyển đổi img_contours sang định dạng mà Tkinter có thể sử dụng
    img_contours_rgb = cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)  # Chuyển đổi từ BGR sang RGB
    img_contours_pil = Image.fromarray(img_contours_rgb)  # Chuyển đổi sang PIL Image
    img_contours_pil = img_contours_pil.resize((400, 150))  # Resize để hiển thị rõ hơn

    # Cập nhật cropped_plate_label với hình ảnh contours
    cropped_img_tk = ImageTk.PhotoImage(img_contours_pil)
    cropped_plate_label.config(image=cropped_img_tk, width=400, height=150)
    cropped_plate_label.image = cropped_img_tk  # Giữ tham chiếu

    return ''.join(predicted_numbers)

def find_license_plate():
    if selected_image_path:  # Kiểm tra xem đã có ảnh được chọn chưa
        # Tải ảnh bằng OpenCV để xử lý với YOLO
        image = cv2.imread(selected_image_path)
        model = YOLO('runs/detect/train4/weights/best.pt')
        results = model(image)  # Dự đoán đối tượng trong ảnh

        for r in results:
            boxes = r.boxes
            im_array = r.plot(line_width=3, font_size=2)  # Tạo hình ảnh với kết quả dự đoán
            im = Image.fromarray(im_array[..., ::-1])  # Chuyển đổi từ mảng NumPy sang hình ảnh PIL

            # Resize ảnh theo kích thước của image_label (400x300)
            im = im.resize((400, 300))

            # Chuyển hình ảnh thành định dạng mà Tkinter có thể hiển thị
            img_tk = ImageTk.PhotoImage(im)

            # Cập nhật Label để hiển thị ảnh đã tìm biển số
            image_label.config(image=img_tk)
            image_label.image = img_tk  # Giữ tham chiếu để không bị xóa bởi Garbage Collector
            for box in boxes:
                # Lấy tọa độ hộp giới hạn của biển số
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Cắt ảnh biển số xe
                cropped_image = image[y1:y2, x1:x2]

                # Lưu ảnh đã cắt
                cv2.imwrite('cropped_license_plate.jpg', cropped_image)

                # Hiển thị ảnh đã cắt trong giao diện Tkinter
                cropped_img = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                cropped_img = cropped_img.resize((400, 150))  # Resize để hiển thị rõ hơn
                cropped_img_tk = ImageTk.PhotoImage(cropped_img)
                cropped_plate_label.config(image=cropped_img_tk, width=400, height=150)
                cropped_plate_label.image = cropped_img_tk  # Giữ tham chiếu

                # Nhận diện ký tự từ ảnh biển số đã cắt
                result = extract_and_predict_numbers('cropped_license_plate.jpg')
                license_plate_label.config(text=f"Biển số xe: {result}")  # Cập nhật nhãn với kết quả

    else:
        # Nếu chưa chọn ảnh, hiển thị thông báo yêu cầu thêm ảnh
        messagebox.showwarning("Cảnh báo", "Vui lòng thêm ảnh trước khi tìm kiếm biển số.")

# Khởi tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Giao diện nhận diện biển số xe")
root.geometry("800x600")

# Nhãn "Nhóm 05" ở trên cùng và giữa
title_label = tk.Label(root, text="Nhóm 05", font=("Arial", 24), fg="blue")
title_label.grid(row=0, column=0, columnspan=2, pady=20)

# Bố cục chia cửa sổ thành 2 phần, trái (40%) và phải (60%)
right_frame = tk.Frame(root)
right_frame.grid(row=1, column=1, sticky="n")

choose_button = tk.Button(right_frame, text="Chọn ảnh", command=choose_image, width=15, height=2, font=("Arial", 14))
choose_button.pack(anchor="w", pady=10)

find_button = tk.Button(right_frame, text="Tìm biển số xe", command=find_license_plate, width=15, height=2,
                        font=("Arial", 14))
find_button.pack(anchor="w", pady=10)

# Ngày giờ thêm ảnh
date_time_label = tk.Label(right_frame, text="Ngày giờ thêm: ", font=("Arial", 14))
date_time_label.pack(anchor="w", pady=10)

# Hiển thị chi tiết các ký tự biển số xe
license_plate_label = tk.Label(right_frame, text="Biển số xe: ", font=("Arial", 14))
license_plate_label.pack(anchor="w", pady=10)

# Phần bên trái (60%): Khu vực hiển thị hình ảnh
left_frame = tk.Frame(root)
left_frame.grid(row=1, column=0, sticky="n", padx=20)

# Khu vực hiển thị hình ảnh gốc
image_label = tk.Label(left_frame, text="Chưa có ảnh", bg="gray", width=60, height=20)
image_label.pack(pady=10)

# Khu vực hiển thị biển số đã cắt ra
cropped_plate_label = tk.Label(left_frame, text="Biển số đã cắt", bg="lightgray", width=60, height=10)
cropped_plate_label.pack(pady=10)

# Chạy vòng lặp chính
root.mainloop()
