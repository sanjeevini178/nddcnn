from flask import Flask, render_template, request, redirect, url_for
# from tensorflow.keras.models import load_model
import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications.vgg16 import preprocess_input
import os
# from tensorflow.keras.preprocessing import image
from PIL import Image
from transformers import pipeline
from pathlib import Path
import uuid
import pymongo

app = Flask(__name__)
# model = load_model('model.h5')
pipe = pipeline("image-classification", "gianlab/swin-tiny-patch4-window7-224-finetuned-parkinson-classification")
# target_img = os.path.join(os.getcwd() , 'static/images')


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
            predictions = pipe(image)

            # Delete the temporary image file
            image_path.unlink()

            # Format the predictions
            formatted_predictions = []
            maxi = 0
            maxi_label = ""
            for prediction in predictions:
                label = prediction['label']
                score = prediction['score']
                if score > maxi:
                  maxi = score
                  maxi_label = label
                formatted_prediction = f"{label} => 'score': {score}"
                formatted_predictions.append(formatted_prediction)


            # Apply conditional formatting for 'parkinson' label
            for i, prediction in enumerate(predictions):
                if prediction['label'] == 'parkinson' and prediction['score'] > 0.5:
                    formatted_predictions[i] = f"<span style='color:red'>{formatted_predictions[i]}</span>"

            
            return render_template('result.html',disease=maxi_label,prob=maxi,user_image=temp_image_path, img_name=file.filename, acc_name=acc.filename)
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