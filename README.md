#Object Bounding box Detection

####Flipkart Grid 2018 Hackathon Project

###AIM
- Using deep learning approach, to build a convolution neural network which will give the bounding box of object present in input image.
- The model should work for any flipkart product given in dataset.

###ABOUT
- Tried various model architectures
- Simple modular code for easy managemet
- Used *Keras* - deep learning framework of python
- Each experiment folder contains different CNN model architectures
- Each folder contains
>  **Model.py** - Contains the main CNN model architecture
> **hyperparameter.py **- Contains the respective hyperparameters of CNN model
> **data_loader.py **- Loads the image dataset into main memory in batches for training and testing
>**output_gen.py** - Generates the output and stores it in csv file.
>**plot.py** - Can be used to see the output
>**train.py** - Where all other utilities are called and model is trained. Trained model is saved per epoch. This is the main file which we run before output gen and ploting.

###NOTE
- As the image dataset was around 14GB, it was not possible to upload it on github.
- As I tried various models, there were too many saved models. As model size is huge I couldn't upload it on github.
