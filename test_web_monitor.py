
# test_web_monitor.py
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "NAO 教学系统监控测试"

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)