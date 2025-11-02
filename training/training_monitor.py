#!/usr/bin/env python3
"""
Training Monitor API with UI
Visualizes Antiks-v1 active learning: teacher questions, student answers, corrections
"""
from flask import Flask, render_template, jsonify, request
import json
import os
from pathlib import Path
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Configure Jinja2 to use different delimiters to avoid conflicts with Vue.js
app.jinja_env.variable_start_string = '{$'
app.jinja_env.variable_end_string = '$}'

# Training state
training_state = {
    "status": "idle",  # idle, running, completed
    "iteration": 0,
    "current_challenge": 0,
    "total_challenges": 0,
    "history": [],
    "stats": {
        "total_questions": 0,
        "correct_answers": 0,
        "training_examples": 0,
        "avg_score": 0
    }
}

# File to store training logs
LOG_DIR = Path(__file__).parent / "training_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "training.jsonl"


class TrainingLogger:
    """Logger that writes to file and updates state"""
    
    def __init__(self):
        self.file = None
        self.buffer = []
    
    def log(self, event_type, data):
        """Log an event"""
        timestamp = datetime.now().isoformat()
        entry = {
            "timestamp": timestamp,
            "type": event_type,
            "data": data
        }
        
        # Write to file
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update state based on event type
        if event_type == "challenge":
            training_state["current_challenge"] += 1
            training_state["stats"]["total_questions"] += 1
            training_state["history"].append({
                "iteration": data.get("iteration", 0),
                "challenge": data.get("number", 0),
                "question": data.get("question", ""),
                "status": "pending"
            })
        
        elif event_type == "answer":
            # Find last entry and update
            if training_state["history"]:
                last = training_state["history"][-1]
                last["student_answer"] = data.get("answer", "")
                last["status"] = "answered"
        
        elif event_type == "evaluation":
            # Update last entry with evaluation
            if training_state["history"]:
                last = training_state["history"][-1]
                last["score"] = data.get("score", 0)
                last["correct"] = data.get("correct", False)
                last["feedback"] = data.get("feedback", "")
                last["corrected_sql"] = data.get("corrected_sql", "")
                
                if data.get("correct", False):
                    training_state["stats"]["correct_answers"] += 1
                else:
                    training_state["stats"]["training_examples"] += 1
                
                last["status"] = "evaluated"
                
                # Update avg score
                scores = [h["score"] for h in training_state["history"] if "score" in h]
                if scores:
                    training_state["stats"]["avg_score"] = sum(scores) / len(scores)
        
        elif event_type == "iteration":
            training_state["iteration"] = data.get("iteration", 0)
            training_state["current_challenge"] = 0
            training_state["status"] = "running"
        
        elif event_type == "training":
            training_state["status"] = "training"
        
        elif event_type == "complete":
            training_state["status"] = "completed"


logger = TrainingLogger()


@app.route('/')
def index():
    """Main UI page"""
    return render_template('training_monitor.html')


@app.route('/api/status')
def get_status():
    """Get current training status"""
    return jsonify(training_state)


@app.route('/api/history')
def get_history():
    """Get training history"""
    # Get recent entries
    recent = request.args.get('recent', default=50, type=int)
    history = training_state["history"][-recent:] if recent else training_state["history"]
    return jsonify(history)


@app.route('/api/logs')
def get_logs():
    """Get raw log entries"""
    lines = request.args.get('lines', default=100, type=int)
    
    if not LOG_FILE.exists():
        return jsonify([])
    
    logs = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                pass
    
    # Return last N lines
    return jsonify(logs[-lines:])


@app.route('/api/start', methods=['POST'])
def start_training():
    """Start monitoring (training should be started separately)"""
    data = request.json or {}
    iterations = data.get('iterations', 10)
    
    training_state["status"] = "running"
    training_state["iteration"] = 0
    training_state["current_challenge"] = 0
    training_state["total_challenges"] = iterations * 10
    training_state["history"] = []
    training_state["stats"] = {
        "total_questions": 0,
        "correct_answers": 0,
        "training_examples": 0,
        "avg_score": 0
    }
    
    # Clear old logs
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    
    logger.log("start", {"iterations": iterations})
    
    return jsonify({"status": "started", "iterations": iterations})


@app.route('/api/monitor/log', methods=['POST'])
def log_event():
    """Receive training events from training script"""
    try:
        data = request.json
        event_type = data.get('type')
        event_data = data.get('data', {})
        
        # Log the event
        logger.log(event_type, event_data)
        
        return jsonify({"status": "logged"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset training state"""
    training_state["status"] = "idle"
    training_state["iteration"] = 0
    training_state["current_challenge"] = 0
    training_state["history"] = []
    training_state["stats"] = {
        "total_questions": 0,
        "correct_answers": 0,
        "training_examples": 0,
        "avg_score": 0
    }
    return jsonify({"status": "reset"})


def run_app():
    """Run the Flask app"""
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)


if __name__ == '__main__':
    print("="*70)
    print("🎓 Antiks-v1 Training Monitor")
    print("="*70)
    print("📊 Web UI: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api/status")
    print("="*70)
    print("\nTo start training with monitoring:")
    print("  python training/train_antiks_v1_monitored.py --iterations 10")
    print("\nStarting monitor server...")
    
    run_app()

