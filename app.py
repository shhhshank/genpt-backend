from flask import Flask, request, jsonify, send_from_directory
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import os
import time

from flask_cors import CORS

from ppt_data_gen_gemini import data_gen
from ppt_gen import ppt_gen

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from ppt_to_video import PPTVideoGenerator

app = Flask(__name__)

CORS(app)

ALLOWED_EXTENSIONS = {'pptx', 'ppt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/genpt"
mongo = PyMongo(app)

# File upload configuration
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<path:filename>', methods=['GET'])
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/template/get', methods=['GET'])
def get_templates():
    templates = mongo.db.templates.find()
    return jsonify([{
        "id": str(template["_id"]),
        "thumbnail": template.get("thumbnail"),
        "path": template["path"],
        "has_image": template.get("has_image")
    } for template in templates]), 200


@app.route('/template/get/<template_id>', methods=['GET'])
def get_template_by_id(template_id):
    template = mongo.db.templates.find_one({"_id": mongo.ObjectId(template_id)})
    if not template:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({
        "id": str(template["_id"]),
        "thumbnail": template.get("thumbnail"),
        "path": template["path"],
        "has_image": template.get("has_image")
    }), 200


@app.route('/template/create', methods=['POST'])
def create_template():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    has_image = request.query_string('has_image')
    filename = secure_filename(str(time.time() * 1000) + ".pptx")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ppt_templates' , filename)
    file.save(file_path)

    template = {
        "path": "ppt_templates/" + filename,
        "has_image": has_image if has_image else False
    }

    result = mongo.db.templates.insert_one(template)
    return jsonify({"id": str(result.inserted_id)}), 201

    

@app.route('/context/get', methods=['GET'])
def get_contexts():
    contexts = mongo.db.contexts.find()
    return jsonify([{
        "id": str(context["_id"]),
        "path":context["path"]
    } for context in contexts]), 200


@app.route('/context/get/<context_id>', methods=['GET'])
def get_context_by_id(context_id):
    context = mongo.db.contexts.find_one({"_id": mongo.ObjectId(context_id)})
    if not context:
        return jsonify({"error": "Context not found"}), 404
    return jsonify({
        "id": str(context["_id"]),
        "path":context["path"]
    }), 200


@app.route('/context/create', methods=['POST'])
def create_context():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(str(time.time() * 1000) + ".pdf")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'],'contexts', filename)
    file.save(file_path)

    context = {"path": "contexts/" + filename}
    result = mongo.db.contexts.insert_one(context)
    return jsonify({"id": str(result.inserted_id)}), 201


@app.route('/generate', methods=['POST'])
def generate_ppt():
    payload = request.json
    if 'contextId' in payload:
        path = mongo.db.contexts.find_one({"_id": ObjectId(payload["contextId"])})['path']
        payload['context'] = "uploads/" + path
    
    response = data_gen(payload)
    template_path = mongo.db.templates.find_one({"_id": ObjectId(payload["templateId"])})['path']
    file_path = ppt_gen('uploads/' + template_path, response)
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        clean_file_path = file_path.lstrip('.')
        
        return send_from_directory(
            os.path.dirname(clean_file_path),
            os.path.basename(clean_file_path),
            as_attachment=True,
            download_name=f"{payload.get('title', 'presentation')}.pptx"
        )
    except Exception as e:
        print(f"Error sending file: {str(e)}")
        print(f"Attempted file path: {file_path}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "✗ Error sending presentation file"
        }), 500

@app.route('/convert-ppt-to-video', methods=['POST'])
def convert_ppt_to_video():
    if 'ppt_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['ppt_file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    try:
        # Save uploaded PPT to uploads directory
        filename = secure_filename(file.filename)
        temp_ppt = os.path.join('uploads', 'temp_ppt', filename)
        os.makedirs(os.path.dirname(temp_ppt), exist_ok=True)
        file.save(temp_ppt)
        
        # Initialize video generator
        generator = PPTVideoGenerator()
        
        print("Extracting content from PPT...")
        slides_content = generator.extract_content_from_ppt(temp_ppt)
        
        print("Generating natural script...")
        scripts = generator.generate_script(slides_content)
        
        print("Generating audio for each slide...")
        audio_paths = generator.generate_audio(scripts)
        
        print("Creating final video...")
        video_path = generator.create_video(slides_content, audio_paths)
        
        # Cleanup temporary files
        generator.cleanup()
        os.remove(temp_ppt)
        
        # Send the video file
        return send_from_directory(
            os.path.dirname(video_path),
            os.path.basename(video_path),
            as_attachment=True,
            download_name=f"{os.path.splitext(filename)[0]}_video.mp4"
        )
        
    except Exception as e:
        # Ensure cleanup happens even if there's an error
        try:
            generator.cleanup()
            os.remove(temp_ppt)
        except:
            pass
            
        return jsonify({
            'success': False,
            'error': str(e),
            'message': f'✗ Error generating video: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(port=8080, debug=False)
