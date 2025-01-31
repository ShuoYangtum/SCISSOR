SCISSOR: Mitigating Semantic Bias through Cluster-Aware Siamese Networks for Robust Classification
====


Data sets
----
All of the datasets we used are open-soursed.<br>
Yelp dataset: [https://business.yelp.com/data/resources/open-dataset/](https://business.yelp.com/data/resources/open-dataset/)<br>
GYAFC dataset: [https://adapterhub.ml/explore/formality_classify/gyafc/](https://adapterhub.ml/explore/formality_classify/gyafc/)<br>
Chest-XRay dataset: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)<br>
NotMNIST dataset: [https://www.kaggle.com/datasets/lubaroli/notmnist](https://www.kaggle.com/datasets/lubaroli/notmnist)<br>

Dependencies
----
Before running our code, please ensure that the following dependencies are met.<br> 

| Library  | Version |
| ------------- | ------------- |
| torch  | 2.3.0  |
| torchvision  | 0.19.1  |
| tokenizers  | 0.19.1  |
| transformers  | 4.40.1  |
| spacy  | 3.7.4  |
| shap  | 0.46.0  |
| sentence-transformers  | 3.0.1  |
| scikit-learn  | 1.4.2  |
| markov-clustering  | 0.0.6  |

Running
----
To run our program, you can simply execute the siamese.py file located in the root directory.<br> 

The directory of the files and some commonly used hyperparameters can be passed via the command line.<br> 

Please note that hyperparameters used during training need to be manually adjusted by modifying the relevant sections of the siamese.py code.<br> 

