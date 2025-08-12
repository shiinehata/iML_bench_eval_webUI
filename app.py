from flask import Flask, render_template, jsonify
import datetime

# Khởi tạo Flask app
app = Flask(__name__)

# Route cho trang chủ
@app.route('/')
def home():
    # Render file index.html từ thư mục templates
    return render_template('index.html')

# Route cho một API đơn giản
@app.route('/api/time')
def get_current_time():
    # Trả về thời gian hiện tại dưới dạng JSON
    now = datetime.datetime.now()
    time_data = {
        'message': 'Dữ liệu từ backend Python!',
        'timestamp': now.isoformat()
    }
    return jsonify(time_data)

# Chạy app khi file được thực thi trực tiếp
if __name__ == '__main__':
    # Chạy trên tất cả các địa chỉ IP của máy, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)