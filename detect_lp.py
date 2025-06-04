import cv2
from ultralytics import YOLO

# Tải mô hình YOLO đã huấn luyện
model = YOLO('runs/detect/train4/weights/best.pt')

# Đọc hình ảnh từ đường dẫn
image_path = 'ky-hieu-bien-so-xe-may-tphcm-la-bao-nhieu_1503230242.jpg'  # Đường dẫn đến hình ảnh của chiếc xe
image = cv2.imread(image_path)

# Dự đoán các đối tượng trong hình ảnh
results = model(image)

# Lấy thông tin các đối tượng đã phát hiện
boxes = results[0].boxes  # Các box phát hiện đầu tiên

for box in boxes:
    # Lấy tọa độ của box
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Chuyển đổi tọa độ sang kiểu int

    # Cắt hình ảnh biển số
    cropped_image = image[y1:y2, x1:x2]

    # Lưu hình ảnh biển số đã cắt
    cv2.imwrite('image_test/cropped_license_plate.jpg', cropped_image)

    # Hiển thị hình ảnh đã cắt
    cv2.imshow("Cropped License Plate", cropped_image)
    cv2.waitKey(0)  # Nhấn phím bất kỳ để đóng cửa sổ
    cv2.destroyAllWindows()


#
# model = YOLO('runs/detect/train4/weights/best.pt')
#         results = model(selected_image_path)  # Xử lý ảnh đã chọn bằng mô hình YOLO
#
#         for r in results:
#             print(r.boxes)
#             im_array = r.plot(line_width=3, font_size=2)  # Tạo hình ảnh với kết quả dự đoán
#             im = Image.fromarray(im_array[..., ::-1])  # Chuyển đổi từ mảng NumPy sang hình ảnh PIL
#
#             # Resize ảnh theo kích thước của image_label (400x300)
#             im = im.resize((400, 300))
#
#             # Chuyển hình ảnh thành định dạng mà Tkinter có thể hiển thị
#             img_tk = ImageTk.PhotoImage(im)
#
#             # Cập nhật Label để hiển thị ảnh đã tìm biển số
#             image_label.config(image=img_tk)
#             image_label.image = img_tk  # Giữ tham chiếu để không bị xóa bởi Garbage Collector