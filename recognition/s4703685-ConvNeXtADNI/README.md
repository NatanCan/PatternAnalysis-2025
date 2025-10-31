# COMP3710 Report Code
## Introduction
This project aims to implement a vision model to classify brain MRI data from the Alzheimer's Disease Neuroimaging Initiative (ADNI), and catagorising them as either having Alzheimer's Disease (AD) or Normal Cognitive functions (NC).

In this git repository, I hope to demonstrate a vision model that can classify Alziehmers disease (normal and AD) of the ADNI brain data with an accuracy of 0.8. The vision model implemented to achieve this task is the ConvNeXt model.

## Background information
### What is a ConvNeXt model?
A ConvNeXt model is a convolution model that has been modified by including the architecture of Vision transformers (ViTs). Before the inception of the ConvNeXt model, ViTs outperformed convolution models in all computer vision tasks. However, inexchange for this improved performance, the transformers were computationally expensive and required large data sets for valid training. The idea of the ConvNeXt model was to create a convolution model that could perform as well or better than ViTs, while also retaining the the efficiency and inductive biases of traditional convolution models, achieving the best of both worlds.

### What problem does this solve?
In the context of this problem, identifying Alzheimer’s disease and other medical datasets, models like ConvNeXt are significant as:
1. Medical datasets tend to be small and domain-specific, which means that ViTs may not be able to train with the amount of available data and traditional convolution models may not be accurate/precise enough for practical use
2. Models must be efficient, stable and interpertable.

ConvNeXt models provides the transformer-level accuracy needed for subtle pattern detection like brain images or medical use; without requiring massive data or compute.

### How are ConvNeXt models made?
As mentioned in the What is a ConvNeXt model section, this model began by creating a ResNet model, a CNN architecture, as the base/starting point of the ConvNeXt model. This entailed creating a ConvNormAct() and BottleNeck() class. The ConvNormAct() is the standard building block in many CNN models. The BottleNeck() class acts as a channel expansion and compression mechanism that enables efficient feature transformation while maintaining computational efficiency. 

For the ConvNeXt model the BottleNeck() class is inverted, so instead of the regular structure as seen in the image below:
![alt text](image.png)
the BottleNeck() goes from narrow->wide->narrow; this change in design makes it so that less information is loss in early layers as the input information is not being compressed down which may result in the information being lost. It also allows for better Gradient Flow which reduced gradient vanishing in deep networks and usually makes the training more stable and follows the general architecture of a ViTs' Multi-Layer preceptron.

## Dependencies and Requirements
These are the Libraries/packages that are required to run the files:

torch>=1.13.0

torchvision>=0.14.0

numpy>=1.21.0

scikit-learn>=1.0.0

matplotlib>=3.5.0

Pillow>=9.0.0

To install all libraries/packages/dependencies, input the following command into a terminal of sorts, either command prompt, VSCode's terminal or your own choice:

```
pip install torch torchvision numpy matplotlib scikit-learn
```

## Example inputs and outputs

## Results
Epoch [300/300], Training Loss: 0.1893
Training took 0.6 minutes
Epoch [300/300], Validation Loss: 0.5839, Validation Accuracy: 0.7849

 Test Accuracy: 0.6308

Confusion Matrix:
[[2110 2350]
 [ 973 3567]]

Classification Report:
              precision    recall  f1-score   support

      Normal       0.68      0.47      0.56      4460
          AD       0.60      0.79      0.68      4540

    accuracy                           0.63      9000

## Plots and Figures
The split of the data is as follows:

Training: 80%

Validation: 20%

Testing: 100%

This split strategy ensures sufficient training data for convergence while maintaining adequate samples for reliable validation. Since the testing data was in its own file, seperate to the training images, it was not necessary for any split and can test using the full file with no draw backs.

## Justification

