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

