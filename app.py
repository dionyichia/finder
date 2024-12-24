from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
import rag
import utils
from datetime import datetime

UPLOAD_FOLDER = "./data"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = rag.keys.os.getenv('SECRET_KEY', 'default_secret_key') 
socketio = SocketIO(app)

# Initialize vectorstore and workflow
def initialize_rag():
    # Compile workflow, create vectorstore
    vectorstore = rag.index_and_embed_cur_docs()
    workflow = rag.create_graph(vectorstore)
    return {'workflow': workflow.compile(), 'vectorstore': vectorstore}

# Initialize the RAG application
rag_app = initialize_rag()

# Ensure upload directory exists
rag.keys.os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
# Render Home Template
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    
    # If no file is selected, browser also submit an empty part without filename
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    # If file is allowed, save and process it
    if file and allowed_file(file.filename):
        filename = rag.keys.os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        print(filename)
        print(file.filename)
        
        # Add the uploaded file to the vectorstore
        try:
            # Assuming your rag module has a method to add a new document
            rag.add_document_to_DB(rag_app['vectorstore'], file.filename)
            return jsonify({"success": True, "message": f"File {file.filename} successfully uploaded and indexed", "name": file.filename})
        except Exception as e:
            print(e)
            return jsonify({"success": False, "message": f"Error processing file: {str(e)}", "name": file.filename}), 500
    
    return jsonify({"success": False, "message": "File type not allowed", "name": file.filename}), 400

# @app.route("/query", methods=["GET", "POST"])
# def query():
@socketio.on('send_message')
def handle_message(data):
    # if request.method == "POST":
    #     # Send query to script
    #     data = request.get_json()
        question = data.get("message")

        # if not question:
        #     return render_template('index.html', response="Missing Input. Please provide a query.")

        # Generate Response
        input = {"question": question}

        generated_response = ""
        for output in rag_app['workflow'].stream(input):
            for _, value in output.items():
                chunk = value.get("generation", "")
                generated_response += chunk
        try:
            emit('receive_message', {
                'question': question,
                'response': generated_response,
                'timestamp': datetime.now().strftime("%H:%M")
            })
        except Exception as e:
            emit('receive_message', {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.datetime.now().strftime("%H:%M")
            })

        # return jsonify({"response": generated_response})
            
        # except Exception as e:
        #     return jsonify({"response": f"Error processing question: {e}"})

    # # This is for swithing between old and new chats
    # if request.method == "GET":
    #     return jsonify({"response": "GET request received"})

@app.route('/delete/<document_name>', methods=['DELETE'])
def delete_document(document_name):

    collection_name = rag.keys.os.path.splitext(document_name)[0]

    try:
        # Delete corr collection in DB
        rag.drop_collection(rag_app.get('vectorstore'), collection_name)

        # Delete file in local folder
        file_dict = utils.check_endpoint_for_file(input_file=document_name, input_folder=UPLOAD_FOLDER, parsed_folder="./parsed_data")
        input_file_path = file_dict.get(collection_name, "default")[0] # File base name is the same as collection name
        parsed_file_path = file_dict.get(collection_name, "default")[1]

        if rag.keys.os.path.exists(input_file_path):
            if (rag.keys.os.path.exists(parsed_file_path)):
                rag.keys.os.remove(parsed_file_path)
            else:
                raise Exception(f"The file {document_name} does not exist in parsed file folder")
                
            rag.keys.os.remove(input_file_path)
        else:
            raise Exception(f"The file {document_name} does not exist in input file folder")
        


        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
