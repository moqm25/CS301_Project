**Semantic Segmentation of Satellite Imagery Project for CS 301 - 103,**
**Group 10: Kamil Arif and Moiez Qamar (milestone 3)**


### Method:
We were assigned the TPE (Tree-Parzen Estimator) method of hyperparameter optimization to optimize our model with. 


### TPE:
TPE works by finding parameters that correlate with models that have performed better than the rest. Models are split according to some threshold t, where the top t% of models (based on final Loss value) are considered “good” models, and the rest are considered “bad” models. For each parameter, the algorithm calculates the next value by calculating the probability of a model using that parameter being part of the “good” set, and dividing that probability by the probability of a model using that parameter being part of the “bad” set. A value that maximizes the resulting function has the highest probability of being part of a “good” model, and the lowest probability of being part of a “bad” model.

Repeating this process eventually results in the models converging to an optimal point in a relatively low amount of iterations. 


### Theory:
TPE aims to create a probability model for the objective function. This model is then used to select the hyperparameters (hyperparameter optimization HPO) that are the most promising, which are then used to evaluate the objective function. The difference between hyperparameters and model parameters is that hyper parameters are set by the program executer/engineer before the training begins. In essence, hyper parameters can be thought of as the settings of the model, and the objective is to fine tune them. By using Bayes Rule, TRE creates two distributions of the hyper parameters which are different from each other in which (1) one has a threshold value less than the value of the objective function, and (2) second one has a threshold value greater than the value of the objective function. The expected improvement in the model would be the ratio of the (2)/(1) - since the objective is to maximize the expected improvement, the ratio should be maximized. TPE takes sample hyper parameters from (2) and tests it with the expected improvement, and looks for the highest value yielded. The hyper parameters associated with the highest yield are passed to the objective function and, if calculated correctly, will result in a higher value than the value that the expected improvement produced.


### Our Implementation:
We established a search space of 12 learning rates ranging from 0.01 to 2.5e-5, and 6 filter count factors ranging from 16 to 6. The filter count factor determines how many filters are used in the model, while still maintaining the same proportion of filters between layers as in the original UNet. This allows us to optimize the structure of the model, and ideally rule out overcomplicated models.




  Precision vs Recall Curve:
![Precision vs Recall](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/Precision%20vs%20Recall.png)


  Loss vs. Epochs (0-40 epochs):
![Loss vs. Epochs (0-40 epochs)](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/Training%20and%20Validation%20Loss.png)
  
  ------------------------------------------------------------
  ![image 1](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%301.png)
  ![image 2](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%302.png)  
  ![image 3](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%303.png)
  ![image 4](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%304.png)
  ![image 5](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%305.png)
  ![image 6](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%306.png)
  ![image 7](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%307.png)
  ![image 8](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%308.png)
  ![image 9](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%309.png)
  ![image 10](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%3010.png)
  ![image 11](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%3011.png)
  ![image 12](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%3012.png)
  ![image 13](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%3013.png)
  ![image 14](https://github.com/moqm25/CS301_Project/blob/milestone-3/images/image%3014.png)
