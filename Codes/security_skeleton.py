from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from cryptography.fernet import Fernet
import os

app = Flask(__name__)

# Configuration for JWT
app.config['JWT_SECRET_KEY'] = 'super-secret-key'  # Change this in production
jwt = JWTManager(app)

# Generate a key for encryption
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# In-memory user data
users = {
    "user1": {"password": "password1"},
}

# Endpoint to authenticate users and return a JWT
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    if username not in users or users[username]['password'] != password:
        return jsonify({"msg": "Bad username or password"}), 401

    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token)

# Secure endpoint that requires authentication
@app.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    current_user = get_jwt_identity()
    message = request.json.get('message', '')
    
    # Encrypt the message
    encrypted_message = cipher_suite.encrypt(message.encode())
    
    # Decrypt the message (for demonstration)
    decrypted_message = cipher_suite.decrypt(encrypted_message).decode()
    
    response = f"Received your message, {current_user}. You said: {decrypted_message}"
    return jsonify({"response": response})

# Endpoint for users to get their data (example of data minimization)
@app.route('/userdata', methods=['GET'])
@jwt_required()
def get_userdata():
    current_user = get_jwt_identity()
    # Only return minimal necessary data
    return jsonify({"username": current_user})

if __name__ == '__main__':
    app.run(ssl_context='adhoc')
