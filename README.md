                                Mô hình Nhận Diện Biển Số Xe
1. Chức năng của Sản phẩm
Chọn ảnh đầu vào: Cho phép người dùng tải lên một ảnh xe có chứa biển số để xử lý.

Nhận diện biển số (YOLOv8): Sử dụng mô hình YOLOv8 để xác định vùng chứa biển số trên ảnh.

Nhận diện ký tự (SVM): Cắt biển số ra từ ảnh và sử dụng mô hình phân loại SVM để nhận diện từng ký tự trên biển số.

Hiển thị kết quả: Trên giao diện hiển thị ảnh gốc, vùng biển số đã được cắt, và kết quả nhận diện là chuỗi ký tự biển số.

Thời gian xử lý: Hiển thị ngày giờ ảnh được thêm vào để theo dõi lịch sử thao tác.

2. Yêu cầu Phần cứng và Phần mềm
Phần cứng
CPU: Tối thiểu Intel i3 hoặc tương đương

RAM: 4GB trở lên

Phần mềm
Python: Phiên bản 3.8 hoặc mới hơn

Thư viện cần cài:

ultralytics (YOLOv8)

opencv-python

joblib

Pillow

tkinter

3. Cách sử dụng
Khởi chạy giao diện:

Chọn ảnh: Bấm “Chọn ảnh” và tải lên một ảnh có chứa biển số xe.

Nhận diện: Bấm “Tìm biển số xe”. Ứng dụng sẽ tự động:

Xác định vị trí biển số bằng YOLOv8

Cắt vùng biển số và nhận dạng ký tự bằng SVM

Hiển thị kết quả lên giao diện

4. Các Thông tin Khác

Mô hình YOLOv8 đã được huấn luyện và lưu tại: runs/detect/train4/weights/best.pt

Mô hình SVM huấn luyện để nhận diện ký tự từ ảnh nhị phân: model/svm_model.pkl

