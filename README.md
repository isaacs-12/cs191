# TumorViz
Tumor Identification and Modeling Engine (https://www.TumorViz.com/)

Model compiled with 50 epochs, data scaled to 64x64x64. 
Using Tensorflow 1.14 and Keras 2.2.5 since that's where I figured out custom loss functions first

Files:
  - 'AutoencoderModelBuild.py' (trains the VAE model and saves it to a file)
  - 'model2.py' (the code for the model that is trained in Autoencoder.py)
  - 'GenerateSmallerDataMatrices_2D.ipynb' and 'Model_Train_2D.ipynb' (Earlier code that was written based on my first approach to segmenting the tumors)
  - 'Update_Scans_Index.py' (placeholder not currently functional, intended to manage the Google Drive files automatically)
    * Due to time spent getting the model working this is currently managed manually
  - 'email_send.py' (send the email to the user)
  - 'Project Demo Presentation.pptx' presentation slide deck from demo on 11/20/2020
  
Segmentation Models:
  - Two trained models from 11/19/2020 -- includes limitations mentioned in slide deck
    * These will be updated over time as we can improve accuracy
    * https://www.dropbox.com/sh/x02ycytt1phzv9x/AABTaWi-J9O3wpOTAwaLjEZLa?dl=0

3D Models:
  - Browser-viewable model (test_model.html)
  - Generic model (.stl) for use in 3D modeling softwares or printing
