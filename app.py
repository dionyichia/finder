from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
import rag
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
            return jsonify({"status": "success", "message": f"File {file.filename} successfully uploaded and indexed"})
        except Exception as e:
            print(e)
            return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 500
    
    return jsonify({"status": "error", "message": "File type not allowed"}), 400

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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

