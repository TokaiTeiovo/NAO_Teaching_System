# web_interface/app.py
import json
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import threading

from ai_server.utils.config import Config
from web_interface.modules.data_monitor import DataMonitor
from web_interface.modules.emotion_monitor import EmotionMonitor
from web_interface.modules.system_monitor import SystemMonitor
from web_interface.api.routes import register_api_routes

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'nao_teaching_secret_key'
app.config['JSON_AS_ASCII'] = False
socketio = SocketIO(app, cors_allowed_origins="*")

# Load configuration
config = Config()

# Create monitoring modules
data_monitor = DataMonitor(config)
emotion_monitor = EmotionMonitor(config)
system_monitor = SystemMonitor(config)

# Global data storage
emotion_history = {
    "timestamp": [],
    "emotion": [],
    "emotions": {},
    "learning_states": {}
}

system_metrics = {
    "cpu": [],
    "memory": [],
    "response_time": [],
    "timestamp": []
}

# Maximum history records
MAX_HISTORY = 100

# Register API routes
register_api_routes(app, data_monitor, system_monitor)


# Define main routes
@app.route('/')
def index():
    """Dashboard page"""
    return render_template('dashboard.html')


@app.route('/monitor')
def monitor():
    """Monitoring page"""
    return render_template('monitor.html')


@app.route('/simulator')
def simulator():
    """Simulator page"""
    return render_template('simulator.html')


@app.route('/knowledge')
def knowledge():
    """Knowledge graph visualization page"""
    return render_template('knowledge.html')


@app.route('/sessions')
def sessions():
    """Session history page"""
    all_sessions = data_monitor.get_sessions()
    return render_template('sessions.html', sessions=all_sessions)


@app.route('/session/<session_id>')
def session_details(session_id):
    """Session details page"""
    session_data = data_monitor.get_session_data(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('sessions'))
    return render_template('session_details.html', session=session_data)


@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html', config=config)


@app.route('/api/emotion/history')
def emotion_history_api():
    """Get emotion history data"""
    return jsonify(emotion_history)


@app.route('/api/system/metrics')
def system_metrics_api():
    """Get system metrics data"""
    return jsonify(system_metrics)


@app.route('/api/logs')
def system_logs_api():
    """Get system logs"""
    service = request.args.get('service', 'all')
    lines = int(request.args.get('lines', 50))
    return jsonify(system_monitor.get_logs(service, lines))


@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Send message to AI server"""
    data = request.get_json()
    message = data.get('message', '')

    if not message:
        return jsonify({"status": "error", "message": "Message cannot be empty"})

    # Here you should integrate with the AI server
    # For demonstration, we'll return a simulated response
    response = {
        "status": "success",
        "message": f"Received message: {message}",
        "response": f"NAO: I received your message: \"{message}\""
    }

    # Log the message
    data_monitor.log_message('student', message)

    return jsonify(response)


@app.route('/api/start_process', methods=['POST'])
def start_process():
    """Start a system process"""
    data = request.get_json()
    process_name = data.get('process', '')

    result = system_monitor.start_process(process_name)
    return jsonify(result)


@app.route('/api/stop_process', methods=['POST'])
def stop_process():
    """Stop a system process"""
    data = request.get_json()
    process_name = data.get('process', '')

    result = system_monitor.stop_process(process_name)
    return jsonify(result)


@app.route('/api/restart_process', methods=['POST'])
def restart_process():
    """Restart a system process"""
    data = request.get_json()
    process_name = data.get('process', '')

    result = system_monitor.restart_process(process_name)
    return jsonify(result)


@app.route('/api/export_session/<session_id>')
def export_session(session_id):
    """Export session data"""
    format_type = request.args.get('format', 'json')

    data = data_monitor.export_session_data(session_id, format_type)
    if not data:
        return jsonify({"status": "error", "message": "Session not found"})

    if format_type == 'json':
        return jsonify(json.loads(data))
    elif format_type == 'csv':
        from flask import Response
        return Response(
            data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=session_{session_id}.csv"}
        )


# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print('Client connected')
    emit('server_response', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print('Client disconnected')


@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start monitoring"""
    global monitoring_thread, monitoring_active
    monitoring_active = True
    if not monitoring_thread or not monitoring_thread.is_alive():
        monitoring_thread = threading.Thread(target=monitoring_task)
        monitoring_thread.daemon = True
        monitoring_thread.start()
    emit('server_response', {'data': 'Monitoring started'})


@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop monitoring"""
    global monitoring_active
    monitoring_active = False
    emit('server_response', {'data': 'Monitoring stopped'})


@socketio.on('simulate_emotion')
def handle_simulate_emotion(data):
    """Simulate emotion data"""
    emotion = data.get('emotion', '中性')
    emotion_data = emotion_monitor.generate_emotion_data(emotion)
    update_emotion_history(emotion_data)
    socketio.emit('emotion_update', emotion_data)


@socketio.on('student_message')
def handle_student_message(data):
    """Handle student message from simulator"""
    text = data.get('text', '')
    if not text:
        return

    # Log the message
    data_monitor.log_message('student', text)

    # Simulate NAO response
    import random
    response_time = random.uniform(0.5, 2.0)
    time.sleep(response_time)

    # Generate response based on text
    if "你好" in text or "hello" in text.lower():
        response = "你好！我是NAO助教，有什么我可以帮助你的吗？"
        actions = ["greeting"]
    elif "谢谢" in text or "thanks" in text.lower():
        response = "不客气！很高兴能帮到你。"
        actions = []
    elif "再见" in text or "goodbye" in text.lower():
        response = "再见！如果有问题随时来找我。"
        actions = ["greeting"]
    elif "什么是" in text or "定义" in text:
        # Extract the concept being asked about
        concept = text.replace("什么是", "").replace("?", "").replace("？", "").strip()
        response = f"{concept}是计算机科学中的一个重要概念，它通常用于..."
        actions = ["explaining"]
    else:
        response = "我理解你的问题。让我思考一下..."
        actions = ["thinking"]

    # Log the response
    data_monitor.log_message('NAO', response)

    # Send response
    socketio.emit('system_response', {
        'text': response,
        'actions': actions
    })


# Monitoring task
monitoring_active = False
monitoring_thread = None


def monitoring_task():
    """Monitoring task thread"""
    global monitoring_active
    monitoring_active = True

    while monitoring_active:
        try:
            # Get system metrics
            metrics = system_monitor.get_metrics()
            update_system_metrics(metrics)
            socketio.emit('system_update', metrics)

            # Simulate emotion update (in a real application, this would come from the AI server)
            if emotion_monitor.should_update():
                emotion_data = emotion_monitor.get_current_emotion()
                update_emotion_history(emotion_data)
                socketio.emit('emotion_update', emotion_data)

            # Get and emit logs periodically
            logs = system_monitor.get_logs('all', 10)
            socketio.emit('log_update', logs)

            time.sleep(1)  # Update frequency
        except Exception as e:
            print(f"Monitoring task error: {e}")
            time.sleep(5)  # Wait after error


def update_emotion_history(emotion_data):
    """Update emotion history data"""
    global emotion_history

    # Add timestamp
    current_time = time.time()
    emotion_history["timestamp"].append(current_time)
    emotion_history["emotion"].append(emotion_data.get("emotion", "中性"))

    # Add emotion strengths
    emotions = emotion_data.get("emotions", {})
    for emotion, strength in emotions.items():
        if emotion not in emotion_history["emotions"]:
            emotion_history["emotions"][emotion] = []

        # Fill missing values
        while len(emotion_history["emotions"][emotion]) < len(emotion_history["timestamp"]) - 1:
            emotion_history["emotions"][emotion].append(0)

        emotion_history["emotions"][emotion].append(strength)

    # Add learning states
    learning_states = emotion_data.get("learning_states", {})
    for state, value in learning_states.items():
        if state not in emotion_history["learning_states"]:
            emotion_history["learning_states"][state] = []

        # Fill missing values
        while len(emotion_history["learning_states"][state]) < len(emotion_history["timestamp"]) - 1:
            emotion_history["learning_states"][state].append(0)

        emotion_history["learning_states"][state].append(value)

    # Limit history size
    if len(emotion_history["timestamp"]) > MAX_HISTORY:
        emotion_history["timestamp"] = emotion_history["timestamp"][-MAX_HISTORY:]
        emotion_history["emotion"] = emotion_history["emotion"][-MAX_HISTORY:]

        for emotion in emotion_history["emotions"]:
            emotion_history["emotions"][emotion] = emotion_history["emotions"][emotion][-MAX_HISTORY:]

        for state in emotion_history["learning_states"]:
            emotion_history["learning_states"][state] = emotion_history["learning_states"][state][-MAX_HISTORY:]


def update_system_metrics(metrics):
    """Update system metrics data"""
    global system_metrics

    current_time = time.time()
    system_metrics["timestamp"].append(current_time)
    system_metrics["cpu"].append(metrics.get("cpu", 0))
    system_metrics["memory"].append(metrics.get("memory", 0))
    system_metrics["response_time"].append(metrics.get("response_time", 0))

    # Limit history size
    if len(system_metrics["timestamp"]) > MAX_HISTORY:
        system_metrics["timestamp"] = system_metrics["timestamp"][-MAX_HISTORY:]
        system_metrics["cpu"] = system_metrics["cpu"][-MAX_HISTORY:]
        system_metrics["memory"] = system_metrics["memory"][-MAX_HISTORY:]
        system_metrics["response_time"] = system_metrics["response_time"][-MAX_HISTORY:]


if __name__ == '__main__':
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="NAO Teaching System Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Start monitoring thread
    monitoring_thread = threading.Thread(target=monitoring_task)
    monitoring_thread.daemon = True
    monitoring_thread.start()

    # Start Flask application
    socketio.run(app, host=args.host, port=args.port, debug=args.debug)