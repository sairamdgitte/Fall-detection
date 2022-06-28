The purpose of this project is to build a Fall Detection through computer vision (using CISCO Meraki Cameras) and visualize the predicitons through CV2 package.

## Procedure:
1. Download the dataset from http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
2. Pre-process the data 
3. Build multiple models (I have used LSTM, ConvLSTM and LRCN)
4. Evaluate the models and pick the best performing model
5. Predict on multiple youtube videos 
6. Visualise the prediction by overlaying the result on the video being test

# Pre-processing the data
The data is stored in frames format. So, the two approaches I will be taking are:
    * for lstm model I will need the keypoints mapped onto the person falling/doing daily activity in the video
        * here each video has different number of frames and none of them are recorded at the same fps (E.g: one video has 160 frames and the other 150 and so on...)
        * Hence, the keypoints collected will also have different frames for each video. Extracting the keypoints and storing them as numpy array with a shape of (30, 400, 258) where 30 is number of videos, 400 is the fps, and 258 are the keypoints of left & right hands and the body pose. 
        * to tackle this, we need to standardise the frames after extracting them (standardise-video-frames.py) to a common fps value : I have standardised it to be 400fps.
    * for convLSTM and LRCN I well need the frames as it is.
        * for each frame, extract the pixel intensity values and normalise it by dividing by 255 (since each pixel ranges from 0 to 255) along with the image height and image width (128x128).
        * I have chosen the image size to be 128x128 since I do not have much compute on my system (16Gb RAM) and neither does colab (12GB of RAM). But one great thing about colab is that it gives you GPU RAM as well and to leverage it I have transformed all the numpy arrays to tensors with the same shape.
        * Again since the fps for each videos varry, I have standardised it to be 200 by appending np.zeros((image height, image width, 3)). 3 is the number of channels the image/video is in (RGB) 
        * The shape of the tensor after preprocessing would be (40, 200, 128, 128, 3) - I restricted myself to use only 20 videos of each category (Falling and not falling)
PS: I used my personal laptop to preprocess the image/video data and pushed it to github so that I can clone it later and use colab compute power to train, evaluate, visualize and repeat, it. 



## **<font style="color:rgb(134,19,348)">Step 4: Implement the ConvLSTM Approach</font>**
In this step, we will implement the first approach by using a combination of ConvLSTM cells. A ConvLSTM cell is a variant of an LSTM network that contains convolutions operations in the network. it is an LSTM with convolution embedded in the architecture, which makes it capable of identifying spatial features of the data while keeping into account the temporal relation. 

<center>
<img src="https://drive.google.com/uc?export=view&id=1KHN_JFWJoJi1xQj_bRdxy2QgevGOH1qP" width= 500px>
</center>


For video classification, this approach effectively captures the spatial relation in the individual frames and the temporal relation across the different frames. As a result of this convolution structure, the ConvLSTM is capable of taking in 3-dimensional input `(width, height, num_of_channels)` whereas a simple LSTM only takes in 1-dimensional input hence an LSTM is incompatible for modeling Spatio-temporal data on its own.

You can read the paper [**Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting**](https://arxiv.org/abs/1506.04214v1) by **Xingjian Shi** (NIPS 2015), to learn more about this architecture.


### **<font style="color:rgb(134,19,348)">Step 4.1: Construct the Model</font>**


To construct the model, we will use Keras [**`ConvLSTM2D`**](https://keras.io/api/layers/recurrent_layers/conv_lstm2d) recurrent layers. The **`ConvLSTM2D`** layer also takes in the number of filters and kernel size required for applying the convolutional operations. The output of the layers is flattened in the end and is fed to the **`Dense`** layer with softmax activation which outputs the probability of each action category. 

We will also use **`MaxPooling3D`** layers to reduce the dimensions of the frames and avoid unnecessary computations and **`Dropout`** layers to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting) the model on the data. The architecture is a simple one and has a small number of trainable parameters. This is because we are only dealing with a small subset of the dataset which does not require a large-scale model.


## **<font style="color:rgb(134,19,348)">Step 5: Implement the LRCN Approach</font>**

In this step, we will implement the LRCN Approach by combining Convolution and LSTM layers in a single model. Another similar approach can be to use a CNN model and LSTM model trained separately. The CNN model can be used to extract spatial features from the frames in the video, and for this purpose, a pre-trained model can be used, that can be fine-tuned for the problem. And the LSTM model can then use the features extracted by CNN, to predict the action being performed in the video. 


But here, we will implement another approach known as the Long-term Recurrent Convolutional Network (LRCN), which combines CNN and LSTM layers in a single model. The Convolutional layers are used for spatial feature extraction from the frames, and the extracted spatial features are fed to LSTM layer(s) at each time-steps for temporal sequence modeling. This way the network learns spatiotemporal features directly in an end-to-end training, resulting in a robust model.

<center>
<img src='https://drive.google.com/uc?export=download&id=1I-q5yLsIoNh2chfzT7JYvra17FsXvdme'>
</center>


You can read the paper [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389?source=post_page---------------------------) by Jeff Donahue (CVPR 2015), to learn more about this architecture.

We will also use [**`TimeDistributed`**](https://keras.io/api/layers/recurrent_layers/time_distributed/) wrapper layer, which allows applying the same layer to every frame of the video independently. So it makes a layer (around which it is wrapped) capable of taking input of shape `(no_of_frames, width, height, num_of_channels)` if originally the layer's input shape was `(width, height, num_of_channels)` which is very beneficial as it allows to input the whole video into the model in a single shot. 

<center>
<img src='https://drive.google.com/uc?export=download&id=1CbauSm5XTY7ypHYBHH7rDSnJ5LO9CUWX' width=400>
</center>


### **<font style="color:rgb(134,19,348)">Step 5.1: Construct the Model</font>**

To implement our LRCN architecture, we will use time-distributed **`Conv2D`** layers which will be followed by **`MaxPooling2D`** and **`Dropout`** layers. The feature extracted from the **`Conv2D`** layers will be then flattened using the  **`Flatten`** layer and will be fed to a **`LSTM`** layer. The **`Dense`** layer with softmax activation will then use the output from the **`LSTM`** layer to predict the action being performed.


## Evaluate the models
LSTM perfomed better than LRCN and ConvLSTM - surprisingly. I believe this could be becasue there's no much data volume to work with. And another reason could be since I am extracting the keypoints for the pure LSTM model and not for the other 2, which creates the difference. 

## Predict on videos
I have resorted to 2 approaches
1. to predict on a single video consisting a single action
2. to predict on multiple actions one after the other in a single video. 

## Future work
If you have the compute and storage, then combing the both approaches mentioned above, i.e., pixel extraction (used for convLSTM and LRCN) and keypoints extraction (used for LSTM) may result in a stronger, more accurate model.