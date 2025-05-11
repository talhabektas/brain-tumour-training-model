# Brain Tumor Classification Project

This project aims to classify brain tumors from MRI images using deep learning techniques. The project includes a comparative analysis using both a custom-designed CNN model and a transfer learning approach (based on VGG16).

## Project Content

The project classifies brain tumors into four different categories:
- Glioma
- Meningioma
- Pituitary
- No Tumor

![Data Samples](data_samples.png)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

You can run the project using the following commands:

### Run all processes in sequence:

```bash
python main.py
```

### Run specific steps:

```bash
python main.py --steps download explore preprocess custom_cnn transfer evaluate
```

Available steps:
- `download`: Checks the existence of the dataset
- `explore`: Performs dataset exploration
- `preprocess`: Performs data preprocessing steps
- `custom_cnn`: Trains the custom CNN model
- `transfer`: Trains the transfer learning model
- `evaluate`: Evaluates the models

## Project Structure

- `main.py`: Main execution file
- `download_data.py`: Checks the existence of the dataset
- `data_exploration.py`: Performs dataset analysis
- `preprocessing.py`: Performs data preprocessing operations
- `custom_cnn.py`: Creates and trains the custom CNN model
- `transfer_learning.py`: Creates and trains the VGG16-based transfer learning model
- `model_evaluation.py`: Evaluates the performance of trained models
- `models/`: Directory where trained models are saved
- `logs/`: Directory where TensorBoard logs are saved
- `plots/`: Directory where graphs are saved
- `data/`: Directory containing the dataset

## Models

### Custom CNN Model

The custom CNN model consists of the following layers:
- 4 Convolutional layers (32, 64, 128, 128 filters)
- MaxPooling layers
- Dropout (0.5) to prevent overfitting
- Fully connected layer with 512 neurons
- Output layer with 4 classes

### Transfer Learning Model

VGG16-based transfer learning model:
- Pre-trained VGG16 base model with ImageNet weights
- Customized top layers (512 neurons, dropout, 4-class output)
- Two-stage training:
  - First stage: Base model frozen
  - Second stage (Fine-tuning): Last layers trainable

## Requirements

```
tensorflow>=2.5.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.2
seaborn>=0.11.1
scikit-learn>=0.24.2
opencv-python>=4.5.2
kaggle>=1.5.12
jupyter>=1.0.0
```

## Results

Training results and model performances are visualized in the `plots/` directory. For model evaluation results, you can run `model_evaluation.py`.

## License

This project is open source. 
