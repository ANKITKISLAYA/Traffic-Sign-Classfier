# Traffic-Sign-Classifier

This repository contains two folders: "TrafficSignClassifierUsingLenetAlexnet" and "TrafficSignClassifierUsingVggInceptionResnet". These folders contain Jupyter Notebooks demonstrating the training and evaluation of traffic sign classification models using different architectures.

## TrafficSignClassifierUsingLenetAlexnet

Inside the "TrafficSignClassifierUsingLenetAlexnet" folder, you will find the following notebooks:

- "TrainingTrafficsignDatasetOnAlexnet.ipynb": Using the AlexNet architecture, a validation accuracy of 0.96 was achieved. The pre-trained weights of AlexNet, obtained from training on the ImageNet dataset, were used. The German Traffic Sign Recognition Benchmark (GTSRB) dataset was used for training and validation. The weights can be obtained from the following link: [Pre-trained AlexNet Weights](https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/).

- "TrainingTrafficsignDatasetOnLeNet.ipynb": Using the LeNet architecture, a validation accuracy of 0.911 was achieved. The implementation details and dataset sources can be found within the notebook.

## TrafficSignClassifierUsingVggInceptionResnet

Inside the "TrafficSignClassifierUsingVggInceptionResnet" folder, you will find the following notebooks:

- "Generate_Bottleneck_Features.ipynb": This notebook utilizes pre-trained CNN models (VGG, Inception, and ResNet) to generate bottleneck features for the traffic dataset. These features, along with the corresponding labels, are saved for further use.

- "VGG_Traffic.ipynb": In this notebook, the fully connected layer is applied on top of the VGG bottleneck features, and the model is trained and evaluated. The achieved accuracy is 0.960.

- "Resnet_Traffic.ipynb": This notebook follows a similar process as the previous one but using the ResNet bottleneck features. The achieved accuracy is 0.933.

- "Inception_Traffic.ipynb": Similarly, this notebook applies the fully connected layer on top of the Inception bottleneck features. The achieved accuracy is 0.961.

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used for training and validation. It can be accessed from the following link:
- [GTSRB Dataset](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Usage

To reproduce the results and train the traffic sign classification models, follow these steps:

1. Download the GTSRB dataset from the provided link and place it in the appropriate folder within the respective notebooks.
2. For the "TrainingTrafficsignDatasetOnAlexnet.ipynb" notebook, download the pre-trained weights of AlexNet from the provided link and save them in the designated folder.
3. Open the desired notebook in Jupyter Notebook or JupyterLab.
4. Execute the cells in the notebook sequentially to perform data preprocessing, model training, and evaluation.
5. Observe the achieved accuracy for the respective architecture.

Please note that the notebooks assume the availability of the required dependencies, including TensorFlow, NumPy, and other common machine learning libraries.

## Credits

I would like to give credit to Udacity for providing the guidance and resources that inspired the development of the "Traffic-Sign-Classfier" project.

## Contact

For any additional questions or inquiries, please feel free to reach out to me at ankitkislaya@gmail.com.

Thank you for exploring the "Traffic-Sign-Classfier"
