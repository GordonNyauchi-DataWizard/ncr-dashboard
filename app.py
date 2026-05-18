from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

@app.route('/')
def index():
    """Render the main NCR Classification page"""
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify_root_cause():
    """
    API endpoint to classify a root cause
    
    Expected JSON:
    {
        "description": "Root cause description text"
    }
    
    Returns:
    {
        "status": "success",
        "message": "Classification successful",
        "data": {
            "categories": [
                {"category": "Category 1", "confidence": 0.95},
                {"category": "Category 2", "confidence": 0.87},
                {"category": "Category 3", "confidence": 0.76}
            ]
        }
    }
    """
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        
        if not description:
            return jsonify({
                "status": "error",
                "message": "Description cannot be empty"
            }), 400
        
        # TODO: Integrate your BERTopic classification model here
        # For now, returning placeholder response
        categories = [
            {"category": "Process Error", "confidence": 0.92},
            {"category": "Human Error", "confidence": 0.85},
            {"category": "Equipment Failure", "confidence": 0.78}
        ]
        
        return jsonify({
            "status": "success",
            "message": "Classification completed successfully",
            "data": {
                "categories": categories,
                "description": description
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error during classification: {str(e)}"
        }), 500

@app.route('/api/corrective-action', methods=['POST'])
def save_corrective_action():
    """
    API endpoint to save corrective actions
    
    Expected JSON:
    {
        "root_cause": "Root cause description",
        "category": "Category",
        "action": "Corrective action description"
    }
    """
    try:
        data = request.get_json()
        root_cause = data.get('root_cause', '').strip()
        category = data.get('category', '').strip()
        action = data.get('action', '').strip()
        
        if not all([root_cause, category, action]):
            return jsonify({
                "status": "error",
                "message": "All fields are required"
            }), 400
        
        # TODO: Save to database
        record = {
            "id": f"NCR_{len([]) + 1}",
            "root_cause": root_cause,
            "category": category,
            "action": action,
            "timestamp": "2024-01-20"
        }
        
        return jsonify({
            "status": "success",
            "message": "Corrective action saved successfully",
            "data": record
        }), 201
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error saving corrective action: {str(e)}"
        }), 500

@app.route('/api/records', methods=['GET'])
def get_records():
    """
    API endpoint to retrieve all saved NCR records
    
    Returns:
    {
        "status": "success",
        "data": [
            {
                "id": "NCR_001",
                "root_cause": "...",
                "category": "...",
                "action": "...",
                "timestamp": "..."
            }
        ]
    }
    """
    try:
        # TODO: Fetch from database
        records = []
        
        return jsonify({
            "status": "success",
            "data": records
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error fetching records: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "NCR Classification API"
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Resource not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
