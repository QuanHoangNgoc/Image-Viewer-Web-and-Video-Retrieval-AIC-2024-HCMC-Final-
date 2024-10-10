from flask import Flask, request, jsonify
from flask_cors import CORS
import back_model 


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/get-local', methods=['POST'])
def send_text():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Check if 'message' is in the received data
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    # Get the text from the request
    message = data['message']
    
    # Process the message (this can be any logic)
    response_message = back_model.get_local(message, matrix_norm, 100)
    print("*", response_message)
    
    # Return the response
    return jsonify({'response': response_message})


if __name__ == '__main__':
    app.run(debug=True)  # Set debug=True for development
