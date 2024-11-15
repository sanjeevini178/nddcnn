from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
from functools import wraps
import numpy as np
import datetime
import os
import io
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline
from pathlib import Path
import uuid
import pymongo
from scipy.stats import skew, kurtosis
import pandas as pd
import gridfs
from preprocessing import preprocess_gait_data
from preprocessing_simtk import preprocess_gait_simtk
from io import BytesIO
from bson import ObjectId
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

scaler = StandardScaler()
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
print(app.secret_key)
# model = load_model('model.h5')
# pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
# target_img = os.path.join(os.getcwd() , 'static/images')
processor = AutoImageProcessor.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
model = AutoModelForImageClassification.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

lstm_model = load_model('train_model.h5', compile=False)

# Connection string
connection_string = "mongodb+srv://sem6:ssn@cluster0.q0hpe8v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Access a specific database
db = client['ndd_prediction']

# Access a specific collection within the database
collection = db['Credentials']

storage = db['storage']

fs = gridfs.GridFS(db)

curr_username = ''
for document in collection.find():
    print(document)

# Helper function for login required
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password = int(password)
        curr_username = username
        if username and password:
            # Query MongoDB for the username and password
            user = collection.find_one({"Username": username, "Password": password})
            
            if user:
                # If user exists, store the user ID in the session and redirect to the index page
                print("success")
                session['username'] = username
                session['user_id'] = str(user['_id'])
                print(f"User ID: {session['user_id']}")
                return redirect(url_for('index'))
    
    # If user does not exist or credentials are incorrect, redirect back to the login page
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            password = int(password)
            # Check if the user already exists
            existing_user = collection.find_one({"Username": username})
            if existing_user is None:
                # Add the new user to the MongoDB collection
                new_user = {"Username": username, "Password": password}
                result = collection.insert_one(new_user)
                user_id = str(result.inserted_id)
                print(f"New user ID: {user_id}")
                return redirect(url_for('login'))
            else:
                error_message = "User already exists. Please try logging in."
                return render_template('signup.html', error=error_message)
    return render_template('signup.html')


@app.route('/index')  # Define the route URL path
@login_required
def index():
    # You can render the index.html template here
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
@login_required
def main():
    return render_template('main.html')

# Allow files with IMGension png, jpg, and jpeg
ALLOWED_IMG = {'jpg', 'jpeg', 'png'}
ALLOWED_FILE = {'csv'}

def allowed_img(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMG

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE

def calculate_statistics(df):
    stats = {}
    for column in df.columns:
        stats[f'{column}Min'] = df[column].min()
        stats[f'{column}Max'] = df[column].max()
        stats[f'{column}Std'] = df[column].std()
        stats[f'{column}Med'] = df[column].median()
        stats[f'{column}Avg'] = df[column].mean()
        stats[f'{column}Skewness'] = skew(df[column])
        stats[f'{column}Kurtosis'] = kurtosis(df[column])
    return stats

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    file = request.files['file']
    acc = request.files['accelerometer']

    if (file and allowed_img(file.filename)) or (acc and allowed_file(acc.filename)): # Checking file format
        image_path = Path('temp.png')
        file.save(image_path)

        # Open and resize the image
        image = Image.open(image_path).resize((200, 200))

        temp_image_filename = str(uuid.uuid4()) + '.jpg'
        temp_image_path = os.path.join('static/temp_images', temp_image_filename)

        # Save the resized image
        image = image.convert('RGB')
        image.save(temp_image_path)

        proc = processor(image, return_tensors="pt")
        with torch.no_grad():
                outputs = model(**proc)
                logits = outputs.logits

                    # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)

                    # Get the predicted class (optional)
                predicted_class = logits.argmax(-1).item()

                # Convert probabilities to a list or other desired format
                probabilities_list = probabilities.squeeze().tolist()

                # Get predicted class
            #predictions = outputs.logits.argmax(-1).item()

            # Delete the temporary image file
        image_path.unlink()

            # Format the predictions
        if predicted_class == 1:
                predicted_class = "Parkinson"
        else:
                predicted_class = "Healthy"
            

        """ acc_data = acc.read()
        acc_buffer = io.BytesIO(acc_data)

        preprocessed_data = preprocess_gait_data(acc_buffer)
        
        df = preprocessed_data

        df = df.drop(df.columns[0], axis=1)
        test_df = calculate_statistics(df)
        test_df = pd.DataFrame.from_dict(test_df, orient='index').T

        testt_X = test_df.values

        testX_scaled = scaler.fit_transform(testt_X)

        X_test_reshaped = testX_scaled.reshape((testX_scaled.shape[0], 1, testX_scaled.shape[1]))

        # Predict with the loaded model (example)
        categories = ['healthy', 'parkinson', 'huntington', 'als']
        label_encoder = LabelEncoder()
        label_encoder.fit(categories)
        label_names = label_encoder.classes_

        # Create a dictionary of label name to their probabilities
        # predicted_probabilities = np.array([0.2, 0.5, 0.1, 0.2])  # Example predicted probabilities
        predicted_probablities = lstm_model.predict(X_test_reshaped)
        lstmpred = np.argmax(predicted_probabilities,axis=1)
        label_to_probability = {label: prob for label, prob in zip(label_names, predicted_probabilities)}
 """
        acc_data = acc.read()
        acc_buffer = io.BytesIO(acc_data)
        df = preprocess_gait_simtk(acc_buffer)
        
        def _read_csv(filepath):
            # Read CSV file without headers
            return pd.read_csv(filepath, header=None)

        # Function to load a single test file
        # This is similar to how training data is loaded but adapted for a single file

        def load_test_file(filepath):
            COLUMN_NAMES = [
                'Patient ID number',
                'Number of total left steps for this trial',
                'Times of peak angular velocity on left shank',
                'Left gait cycle (stride time) durations',
                'Left swing time durations',
                'Left swing angular ranges',
                'Peak shank angular velocity (left)',
                'Freeze index (left accelerometer)',
                'Left leg identifier',
                'Number of total right steps for this trial',
                'Times of peak angular velocity on right shank',
                'Right gait cycle (stride time) durations',
                'Right swing time durations',
                'Right swing angular ranges',
                'Peak shank angular velocity (right)',
                'Freeze index (right accelerometer)',
                'Right leg identifier'
            ]

            COLUMN_INDICES = {
                'peak_shank_angular_velocity_left': 6, 'peak_shank_angular_velocity_right': 14,
                'stride_time_left': 3, 'stride_time_right': 11,
                'swing_angular_range_left': 5, 'swing_angular_range_right': 13,
                'freeze_index_left': 7, 'freeze_index_right': 15,
            }

            # Read and process the CSV file
            df = filepath

            if df.empty:
                print(f"File {filepath} was empty or had an issue.")
                return pd.DataFrame()

            df = df.apply(pd.to_numeric, errors='coerce')
            processed_data = pd.DataFrame({
                        'peak_shank_angular_velocity': df.iloc[:, COLUMN_INDICES['peak_shank_angular_velocity_left']].fillna(
                            df.iloc[:, COLUMN_INDICES['peak_shank_angular_velocity_right']]),
                        'stride_time': df.iloc[:, COLUMN_INDICES['stride_time_left']].fillna(
                            df.iloc[:, COLUMN_INDICES['stride_time_right']]),
                        'swing_angular_range': df.iloc[:, COLUMN_INDICES['swing_angular_range_left']].fillna(
                            df.iloc[:, COLUMN_INDICES['swing_angular_range_right']]),
                        'freeze_index': df.iloc[:, COLUMN_INDICES['freeze_index_left']].fillna(
                            df.iloc[:, COLUMN_INDICES['freeze_index_right']]),
                    })

            # Calculate arrhythmicity and asymmetry
            print(processed_data['stride_time'])
            print(type(processed_data['stride_time']))
            processed_data['arrhythmicity'] = processed_data['stride_time'].diff().abs()
            processed_data['asymmetry'] = (df.iloc[:, COLUMN_INDICES['stride_time_left']] -
                                        df.iloc[:, COLUMN_INDICES['stride_time_right']]).abs()

            # Add previous step's values
            processed_data['prev_peak_shank_angular_velocity'] = processed_data['peak_shank_angular_velocity'].shift(1)
            processed_data['prev_stride_time'] = processed_data['stride_time'].shift(1)

            # Drop NaN values (arising from placeholders or previous steps)
            processed_data.dropna(inplace=True)

            if processed_data.empty:
                print(f"Processed data from {filepath} resulted in an empty DataFrame after NaN removal.")
                return pd.DataFrame()

            return processed_data.values

        # Function to predict using the loaded LSTM model
        def predict_on_test_file(model, test_data):
            if test_data.size == 0:
                print("No valid test data available for prediction.")
                return None
            test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))

            predictions = model.predict(test_data)
            return predictions

    
            # Load the data using the file-like object
        test_data = load_test_file(df)
        predictions = predict_on_test_file(lstm_model, test_data)
        print(predictions)

        average_prediction = np.mean(predictions, axis=0)

        # Get the class with the highest average probability
        collective_prediction = np.argmax(average_prediction)
        swapped_probabilities = {
            '0':'No PD',
            '1':'PD (with FoG)',
            '2':'PD (without FoG)'
        }
        
        swapped2 = {'No PD':average_prediction[0], 'PD (with FoG)':average_prediction[1], 'PD (without FoG)':average_prediction[2]}

        
        # Read the file content into memory before uploading to GridFS
        file_content = file.read()
        acc_content = acc.read()
        file_id = fs.put(file_content, filename=file.filename)
        acc_id = fs.put(acc_content, filename=acc.filename)

        # Get the username from the session
        username = session['username']

        store = {
            'Name': username,
            'Date': datetime.datetime.now(),
            'Drawing': file_id,
            'IMU Data': acc_id,
            'Img Detected': predicted_class,
            'Img probability': max(probabilities_list),
            'Gait Detected': int(collective_prediction),  # Convert to int
    'Gait probability': float(average_prediction[int(collective_prediction)])}  # Convert to float


        storage.insert_one(store)
        
        # Generate PDF report
        pdf_path = f'static/reports/report_{uuid.uuid4()}.pdf'
        generate_pdf(pdf_path, username, predicted_class, max(probabilities_list), collective_prediction, average_prediction[collective_prediction], temp_image_path)


        return render_template('result.html', disease=predicted_class, prob=max(probabilities_list),
                               user_image=temp_image_path, img_name=file.filename, acc_name=acc.filename,
                               gait=swapped2, pdf_report=pdf_path)
    else:
        return "Unable to read the file. Please check file IMGension"

@app.route('/logout')
def logout():
    return render_template('login.html')

@app.route('/history', methods=['GET'])
@login_required
def history():
    username = session.get('username')
    # Access the user's collection
    user_collection = db['storage']
    # Retrieve all records for the user
    user_data = list(user_collection.find({"Name": username}))
    return render_template('history.html', data=user_data)

@app.route('/download/<file_id>')
@login_required
def download_file(file_id):
    try:
        # Convert file_id to ObjectId
        file_id = ObjectId(file_id)
        # Retrieve the file from GridFS using the ObjectId
        file_data = fs.get(file_id)

        # Debug: Check file data size
        file_content = file_data.read()
        print(f"File size: {len(file_content)} bytes")

        # Create a BytesIO object to send the file data
        file_io = BytesIO(file_content)
        # Set the file's name and content type
        filename = file_data.filename
        content_type = file_data.content_type if file_data.content_type else 'application/octet-stream'
        
        # Debug: Log the filename and content type
        print(f"Downloading file: {filename}, Content Type: {content_type}")

        # Send the file to the client
        return send_file(file_io, as_attachment=True, download_name=filename, mimetype=content_type)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 404

def generate_pdf(pdf_path, username, predicted_class, img_prob, gait_class, gait_prob, image_path):
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica", 12)
    c.drawString(100, height - 50, f"Prediction Report for {username}")
    c.drawString(100, height - 70, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, height - 90, f"Image Detected: {predicted_class}")
    c.drawString(100, height - 110, f"Image Probability: {img_prob:.2f}")
    c.drawString(100, height - 130, f"Gait Detected: {gait_class}")
    c.drawString(100, height - 150, f"Gait Probability: {gait_prob:.2f}")

    # Add the image to the PDF
    if os.path.exists(image_path):
        c.drawImage(image_path, 100, height - 400, width=200, height=200)

    c.showPage()
    c.save()

@app.route('/download_report/<path:filename>', methods=['GET'])
@login_required
def download_report(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True, port=8000)
