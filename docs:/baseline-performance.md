**Semantic Segmentation of Satellite Imagery Project for CS 301 - 103,**\
**Group 10: Kamil Arif and Moiez Qamar**\
\
Model\
The model used was the custom model provided in the github from the description of the provided video. We decided not to use the resnet34 option in hopes that the custom model would consume less RAM then loading an entire external model at runtime.\
The model is a convolutional autoencoder following the UNet architecture. Every input image is convolved down to a 256x1x1 layer of filters, before being expanded back out to an image.  \
Training Process\
The initial issue during training was GPU RAM limitations; to solve this problem and get a rudimentary model started, the model was trained on portions of the dataset. The first 40 epochs were trained on all data excluding Tile 7, and the next 40 epochs were trained on all data except Tiles 6 and 8. After that, the model was trained for 20 epochs on Tiles 2, 6, 7 and 8, and 20 epochs on the remaining tiles.\
After the first 120 epochs, the model was trained on the full dataset with a significantly reduced batch size and learning rate, prior to this, it had been training in batches of 16 with the default Adam learning rate of 0.001. First it was trained for 20 epochs with a batch size of 4 and a learning rate of 10<sup>-4</sup>. After that it was trained for approximately 60 epochs with a batch size of 1 and a learning rate of 10<sup>-6</sup>.\
\
Note: Due to the fact that the model was trained in segments, a full graph of Loss over all Epochs is not available. Instead a representative graph is given of what the first 40 epochs looked like, generated from a fresh model. \
\
  Precision vs Recall Curve:\
![Precision vs Recall](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/Precision%20vs%20Recall.png)\
\
\
  Loss vs. Epochs (0-40 epochs):\
![Loss vs. Epochs (0-40 epochs)](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/Training%20and%20Validation%20Loss.png)\
  \
  ------------------------------------------------------------\
  ![image 1](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%201.png)\
  ![image 2](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%202.png)  \
  ![image 3](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%203.png)\
  ![image 4](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%204.png)\
  ![image 5](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%205.png)\
  ![image 6](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%206.png)\
  ![image 7](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%207.png)\
  ![image 8](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%208.png)\
  ![image 9](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%209.png)\
  ![image 10](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%2010.png)\
  ![image 11](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%2011.png)\
  ![image 12](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%2012.png)\
  ![image 13](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%2013.png)\
  ![image 14](https://github.com/moqm25/CS301_Project/blob/milestone-2/images/image%2014.png)}
