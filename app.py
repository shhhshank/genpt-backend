from flask import Flask, request, jsonify, send_from_directory
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import os
import time
import asyncio

from flask_cors import CORS

from ppt_data_gen import data_gen
from ppt_gen import ppt_gen
from img_gen import RealisticImageGenerator

app = Flask(__name__)
image_gen = RealisticImageGenerator()

CORS(app)


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
        "path": template["path"]
    } for template in templates]), 200


@app.route('/template/get/<template_id>', methods=['GET'])
def get_template_by_id(template_id):
    template = mongo.db.templates.find_one({"_id": mongo.ObjectId(template_id)})
    if not template:
        return jsonify({"error": "Template not found"}), 404
    return jsonify({
        "id": str(template["_id"]),
        "thumbnail": template.get("thumbnail"),
        "path": template["path"]
    }), 200


@app.route('/template/create', methods=['POST'])
def create_template():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filename = secure_filename(str(time.time() * 1000) + ".pptx")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ppt_templates' , filename)
    file.save(file_path)

    template = {
        "path": "ppt_templates/" + filename 
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
        
    respone = data_gen(payload)
    template_path = mongo.db.templates.find_one({"_id": ObjectId(payload["templateId"])})['path']
    print(template_path)
    downloadUrl = ppt_gen("uploads/" + template_path, respone)

    return jsonify({"url": downloadUrl}), 200


async def fetch_and_process():
    prompts = []
    
    while True:
        prompt = input("Enter prompt: ")
        if prompt != "-1":
            prompts.append(prompt)
        else:
            break
        
    await image_gen.process_prompts(prompts=prompts)

if __name__ == '__main__':
    asyncio.run(fetch_and_process())
    print("Test for async")
        
    
    

