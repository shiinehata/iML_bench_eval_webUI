# Dockerfile

# 1. Chọn một image Python có sẵn làm nền tảng
FROM python:3.11-slim

# 2. Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# 3. Sao chép file requirements.txt vào container
COPY requirements.txt .

# 4. Cài đặt các thư viện Python đã liệt kê
RUN pip install --no-cache-dir -r requirements.txt

# 5. Sao chép toàn bộ code của bạn vào container
COPY . .

# 6. Mở cổng 5000 để bên ngoài có thể truy cập vào
EXPOSE 5000

# 7. Lệnh sẽ được chạy khi container khởi động
CMD ["python", "app.py"]