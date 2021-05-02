import os
from flask import Flask
from flask import render_template
from flask import request
import re
import os
import base64
import time
import sys
import numpy as np
import uuid
import json
import tensorflow as tf
import requests
from tensorflow.keras.models import Model
from annoy import AnnoyIndex
import os
import io
from flask import request
import cv2
import numpy as np
from PIL import Image
from flask import  jsonify
import tensorflow as tf
from flask import Flask
from flask import render_template

app = Flask(__name__)
UPLOAD_FOLDER="/home/jainsam123/Downloads/newProj/static"
path_to_model = "/home/jainsam123/Downloads/newProj/models/xception_224x224_adam_batch32_8labels_5000each_10ep_ft16ep.h5"
path=path_to_model
model = None
feature_extractor = None
ann_index = []
ann_metadata = []

labels = ['Cell_Phones_and_Accessories', 'Clothing_Men', 'Clothing_Women', 'Electronics', 'Home_and_Kitchen', 'Pet_Supplies', 'Shoes', 'Watches']

model = tf.keras.models.load_model(path)
layer_name = 'global_average_pooling2d_2'
feature_extractor = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

for i in range(len(labels)):
        ann_index_name = 'index_xception_224x224_adam_batch32_8labels_5000each_10ep_ft16ep_label_{}.ann'.format(i)
        ann_metadata_name = 'metadata_xception_224x224_adam_batch32_8labels_5000each_10ep_ft16ep_label_{}.json'.format(i)

        path_ann_index = '/home/jainsam123/Downloads/newProj/models/annoy_index/label separated/' + ann_index_name
        path_ann_metadata = '/home/jainsam123/Downloads/newProj/models/annoy_index/label separated/' + ann_metadata_name

        with open(path_ann_metadata) as f:
            ann_metadata_data = json.load(f)
            ann_metadata.append(ann_metadata_data)

        ann_index_obj = AnnoyIndex(ann_metadata_data['features_length'], metric='angular')
        ann_index_obj.load(path_ann_index)
        ann_index.append(ann_index_obj)


def get_neighbors(label, input_feature_vectors, max_neighbors=14):
    results = []
    
    # get the nearest neighbors of that first nearest neighbor
    ann_index_obj = ann_index[label]

    for item_id in ann_index_obj.get_nns_by_vector(input_feature_vectors, max_neighbors, search_k=10):
        # if item_id != top_1:
        results.append({
        'id': item_id,
        'asin': ann_metadata[label]['list_asin'][item_id]
        })

    # print('get_neighbors label', label)

    return results

def preprocess_image(image_location):
    input_image = Image.open(image_location)
    predict_img_width = 224
    predict_img_height = 224

    img_str = np.array(input_image)

    image=tf.convert_to_tensor(img_str, dtype=tf.float32)
    image = tf.image.resize(image, [predict_img_width, predict_img_height])
    image = (255 - image) / 255.0  # normalize to [0,1] range
    image = tf.reshape(image, (1, predict_img_width, predict_img_height, 3))

    return image


@app.route('/',methods=["GET","POST"])

def upload_predict():

    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            #input_image = Image.open(image_location)
            image = preprocess_image(image_location)
            prediction_probs = model.predict(image)
            predicted_label = np.argmax(prediction_probs, axis=1)[0]
            input_feature_vectors = feature_extractor.predict(image)
            input_feature_vectors = input_feature_vectors.flatten()
            MAX_TOP_K=30
            input_feature_vectors = input_feature_vectors / input_feature_vectors.max()
            top_k = get_neighbors(predicted_label, input_feature_vectors, MAX_TOP_K)
            
            # print(predicted_label)

            for x in range(MAX_TOP_K):
                print (top_k[x]["asin"])




            # print(prediction_probs)
            return render_template("index.html", prediction=2,img_name=top_k , clothType=labels[predicted_label])

    return render_template("index.html", prediction=0 , img_name= [], clothType=None)

if __name__ == '__main__':
   app.run(port=5000,debug=True)  
 
