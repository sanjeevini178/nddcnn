from flask import Flask, render_template, request, redirect, url_for

import numpy as np

import os
import torch
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

for document in collection.find():
    print(document)

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password=int(password)
        
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

# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

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
                '''logits = outputs.logits

                    # Apply softmax to get probabilities
                    probabilities = F.softmax(logits, dim=-1)

                    # Get the predicted class (optional)
                    predicted_class = logits.argmax(-1).item()

                # Convert probabilities to a list or other desired format
                probabilities_list = probabilities.squeeze().tolist()
'''
                # Get predicted class
            predictions = outputs.logits.argmax(-1).item()

            # Delete the temporary image file
            image_path.unlink()

            # Format the predictions
            formatted_predictions = []
            maxi = 0
            maxi_label = ""
            """ for prediction in predictions:
                label = prediction['label']
                score = prediction['score']
                if score > maxi:
                  maxi = score
                  maxi_label = label
                formatted_prediction = f"{label} => 'score': {score}"
                formatted_predictions.append(formatted_prediction) """


            # Apply conditional formatting for 'parkinson' label
            """ for i, prediction in enumerate(predictions):
                if prediction['label'] == 'parkinson' and prediction['score'] > 0.5:
                    formatted_predictions[i] = f"<span style='color:red'>{formatted_predictions[i]}</span>" """

            
            
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
            return render_template('result.html',prob=predictions,user_image=temp_image_path, img_name=file.filename, acc_name=acc.filename, gait=label_to_probability)
            """ filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              fruit = "Apple"
            elif classes_x == 1:
              fruit = "Banana"
            else:
              fruit = "Orange"
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('predict.html', fruit = fruit,prob=class_prediction, user_image = file_path) """
        else:
            return "Unable to read the file. Please check file IMGension"


if __name__ == '__main__':
  
    app.run(debug=True,use_reloader=True, port=8000)