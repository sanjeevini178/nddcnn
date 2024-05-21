from flask import Flask, render_template, request, redirect, url_for, session

import numpy as np
import datetime
import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline
from pathlib import Path
import uuid
import pymongo
from scipy.stats import skew, kurtosis
import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification

scaler=StandardScaler()
app = Flask(__name__)
app.secret_key = os.urandom(24).hex()
print(app.secret_key)
# model = load_model('model.h5')
# pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
# target_img = os.path.join(os.getcwd() , 'static/images')
processor = AutoImageProcessor.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
model = AutoModelForImageClassification.from_pretrained("gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")

lstm_model = load_model('lstm_model.h5',compile=False)

# Connection string
connection_string = "mongodb+srv://sem6:ssn@cluster0.q0hpe8v.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = pymongo.MongoClient(connection_string)

# Access a specific database
db = client['ndd_prediction']

# Access a specific collection within the database
collection = db['Credentials']

storage = db['storage']

curr_username = ''
for document in collection.find():
    print(document)
""" 
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password=int(password)
        curr_username = username
        if username and password:
            # Query MongoDB for the username and password
            user = collection.find_one({"Username": username, "Password": password})
            
            if user:
                # If user exists, render the home page template
                print("sucess")
                return redirect(url_for('index'))
    
    # If user does not exist or credentials are incorrect, redirect back to the login page
    return render_template('login.html')

@app.route('/index')  # Define the route URL path
def index():
    # You can render the index.html template here
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
def main():
    return render_template('main.html')

#Allow files with IMGension png, jpg and jpeg
ALLOWED_IMG = set(['jpg' , 'jpeg' , 'png'])
ALLOWED_FILE = set(['csv'])

def allowed_img(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_IMG
           
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_FILE

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

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        acc = request.files['accelerometer']
        
        if (file and allowed_img(file.filename)) or (acc and allowed_file(acc.filename)): #Checking file format
            image_path = Path('temp.png')
            file.save(image_path)

            # Open and resize the image
            image = Image.open(image_path).resize((200, 200))

            temp_image_filename = str(uuid.uuid4()) + '.jpg'
            temp_image_path = os.path.join('static/temp_images', temp_image_filename)

            # Save the resized image
            image = image.convert('RGB')
            image.save(temp_image_path)


            # Make prediction using the model
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
            if predicted_class==1:
                predicted_class = "Parkinson"
            else:
                predicted_class = "Healthy"
            
            
            df = pd.read_csv(acc)

            df = df.drop(df.columns[0],axis=1)
            test_df = calculate_statistics(df)
            test_df = pd.DataFrame.from_dict(test_df, orient='index').T

            testt_X = test_df.values

            testX_scaled = scaler.fit_transform(testt_X)

            X_test_reshaped = testX_scaled.reshape((testX_scaled.shape[0], 1, testX_scaled.shape[1]))
            # Predict with the loaded model (example)
            lstm_model.compile(RMSprop(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
            test_predictions = lstm_model.predict(X_test_reshaped)
            #pred_list = test_predictions.to_list()
            #diseases = ['healthy','parkinsons','huntington\'s','amylotrophic lateral sclerosis']
            categories = ['healthy', 'parkinson', 'huntington', 'als']
            label_encoder = LabelEncoder()
            label_encoder.fit(categories)
            label_names = label_encoder.classes_

            # Create a dictionary of label name to their probabilities
            predicted_probabilities = test_predictions[0]
            label_to_probability = {label: prob for label, prob in zip(label_names, predicted_probabilities)}

            max_label = max(label_to_probability, key=label_to_probability.get)
            max_probability = label_to_probability[max_label]

            store = {'Name':curr_username, 
                    'Date':datetime.datetime.now(),
                    'Drawing':file, 
                    'IMU Data':acc,
                    'Img Detected':predicted_class,
                    'Img probability':max(probabilities_list),
                    'Gait Detected':max_label,
                    'Gait probability':max_probability
            }

            storage.insert_one(store)
            return render_template('result.html',disease=predicted_class,prob=max(probabilities_list),user_image=temp_image_path, img_name=file.filename, acc_name=acc.filename, gait=label_to_probability)
        else:
            return "Unable to read the file. Please check file IMGension"
 """


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
        password=int(password)
        curr_username = username
        if username and password:
            # Query MongoDB for the username and password
            user = collection.find_one({"Username": username, "Password": password})
            
            if user:
                # If user exists, render the home page template
                print("sucess")
                session['username'] = username
                return redirect(url_for('index'))
    
    # If user does not exist or credentials are incorrect, redirect back to the login page
    return render_template('login.html')

@app.route('/index')  # Define the route URL path
def index():
    # You can render the index.html template here
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/main', methods=['POST'])
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
        if predicted_class==1:
                predicted_class = "Parkinson"
        else:
                predicted_class = "Healthy"
            

        df = pd.read_csv(acc)

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
        predicted_probabilities = np.array([0.2, 0.5, 0.1, 0.2])  # Example predicted probabilities
        label_to_probability = {label: prob for label, prob in zip(label_names, predicted_probabilities)}

        max_label = max(label_to_probability, key=label_to_probability.get)
        max_probability = label_to_probability[max_label]

        store = {'Name': session['username'],
                 'Date': datetime.datetime.now(),
                 'Drawing': file.filename,
                 'IMU Data': acc.filename,
                 'Img Detected': predicted_class,
                 'Img probability': max(probabilities_list),
                 'Gait Detected': max_label,
                 'Gait probability': max_probability
        }

        storage.insert_one(store)

        return render_template('result.html', disease=predicted_class, prob=max(probabilities_list),
                               user_image=temp_image_path, img_name=file.filename, acc_name=acc.filename,
                               gait=label_to_probability)
    else:
        return "Unable to read the file. Please check file IMGension"


if __name__ == '__main__':
  
    app.run(debug=True,use_reloader=True, port=8000)