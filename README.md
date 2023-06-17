# Traffic-Sign-Classfier

This repository contains a Jupyter Notebook named "TrainingTrafficsignDatasetOnAlexnet.ipynb" that demonstrates the training of a traffic sign classification model using the AlexNet and LeNet architectures. The notebook utilizes pre-trained weights of AlexNet obtained from training on the ImageNet dataset and compares the performance of both architectures on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset used for training and validation can be accessed from the following link:
- [GTSRB Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Pre-trained Weights

The pre-trained weights of the AlexNet architecture, which have been fine-tuned on the ImageNet dataset, are used as the starting point for training the traffic sign classification model. These weights can be obtained from the following link:
- [Pre-trained AlexNet Weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/)

## Usage

To reproduce the results and train the traffic sign classification models using the AlexNet and LeNet architectures, follow these steps:

1. Download the GTSRB dataset from the provided link and place it in the appropriate folder within the notebook.
2. Download the pre-trained weights of AlexNet from the provided link and save them in the designated folder.
3. Open the "TrainingTrafficsignDatasetOnAlexnet.ipynb" notebook in Jupyter Notebook or JupyterLab.
4. Execute the cells in the notebook sequentially to perform data preprocessing, model training, and evaluation.
5. Observe the validation accuracy achieved by each model and compare their performance.

Please note that the notebook assumes the availability of the required dependencies, including TensorFlow, NumPy, and other common machine learning libraries.

## Model Performance

- Using LeNet architecture, the model achieved a validation accuracy of 0.911.
- Using AlexNet architecture along with weights obtained from training on the ImageNet dataset, the model achieved a validation accuracy of 0.96.

## Credits

I would like to give credit to Udacity for providing the guidance and resources that inspired the development of the "TrainingTrafficsignDatasetOnAlexnet" project.

## Contact

For any additional questions or inquiries, please feel free to reach out to me at ankitkislaya@gmail.com.

Thank you for exploring the "TrainingTrafficsignDatasetOnAlexnet" repository! I hope the project helps you understand the performance of the AlexNet and LeNet architectures for traffic sign classification, and their respective accuracies on the GTSRB dataset.
