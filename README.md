# Hackfest2021


<h1>Content:</h1>
<a href="#obj" >1. Problem Statement</a><br>
<a href="#pre" >2. Preprocessing </a><br>
<a href="#model" >3. Model</a><br>
<a href="#res" >4. Result</a><br>
<a href="#app" >5. Appendix</a><br>
<hr>
<h1 id="obj">1. Problem Statement</h1>

<strong>
  While watching a movie or a favorite web series on any of the OTT platform, don't you feel the urge that I also want clothes which are similar to the actor /actress, don't you want to look like them or don't you think that I also want to be trendy and look cool among my fellow mates? To develop an app against this that will scan and give similar images compared to the actor and actresses' clothes.
</strong>
<br>


<h1 id="pre">2. Preprocessing:</h1>
We use this python script for preprocessing our data as per our needs.

import os
path = "/home/prince/Documents/Rough/"
for filename in os. listdir(path):
    print(filename)
    filename_without_ext = filename.split("_")[0]
    extension = ".jpg"
    new_file_name = filename_without_ext+extension
    os.rename(path+filename,path+new_file_name)

<h1 id="model">3. Model:</h1>

To classify the images, we used Xception model from Google and used it’s global average pooling layer(second last layer) to extract the feature vector and apply Knn to predict top-k similar feature vector in our database. (In our case we k=30). 

<h1 id="res">4. Result:</h1>
<img src="img/Screenshot from 2021-03-22 11-24-30.png">


