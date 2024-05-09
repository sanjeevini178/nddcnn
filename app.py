from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
import numpy as np
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications.vgg16 import preprocess_input
import os
# from tensorflow.keras.preprocessing import image

app = Flask(__name__)
# model = load_model('model.h5')

# target_img = os.path.join(os.getcwd() , 'static/images')

@app.route('/')
def index_view():
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
            return render_template('result.html',prob="100%",user_image="static/image_1.jpg", img_name=file.filename, acc_name=acc.filename)
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
    app.run(debug=True,use_reloader=False, port=8000)