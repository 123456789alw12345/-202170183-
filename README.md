# Honey Type Classification with TensorFlow

This project leverages **TensorFlow** to build an AI model capable of classifying three types of honey: **Samar**, **Farz**, and **Abyad** based on their images.

## Requirements
- Python 3.6+
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn

## Setup and Usage
1. **Prepare the Dataset**: Organize your images into a `dataset/` directory with three subfolders for each honey type (`samar/`, `farz/`, `abyad/`).
2. **Install Dependencies**:
    ```bash
    pip install tensorflow matplotlib scikit-learn numpy
    ```
3. **Train the Model**: Run the training script to train the CNN model and save the best-performing model.
4. **Classify New Images**: Use the trained model to classify new honey images and save the results to a CSV file.

## Notes
- Ensure balanced datasets for each honey type to improve model accuracy.
- Utilize **Transfer Learning** techniques for enhanced performance.
- Evaluate model performance using accuracy plots and confusion matrices.

## Contributions
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
