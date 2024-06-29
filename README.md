# Project Overview Iris-Flower-Classification :

Iris-Flower-Classification/
├── README.md
├── IrisFlower.ipynb
├── test_api.py
├── app.py
├── data/
│   ├── iris.zip
│   ├── bezdekIris.data
│   ├── iris.data
│   ├── iris.names
├── models/
│   ├── knn_model.pkl
│   ├── log_reg_model.pkl
└── Index

This project involves building and evaluating machine learning models to classify iris flower species based on their characteristics. The project uses the famous Iris dataset and includes the following steps:

1. Load the dataset.
2. Perform exploratory data analysis (EDA) to understand the data distribution.
3. Split the data into training and testing sets.
4. Train two classifiers: Logistic Regression and K-Nearest Neighbors.
5. Evaluate the model performance using metrics like accuracy, precision, recall, and F1-score.
6. Save the trained models.
7. Develop a Flask API to serve the models for prediction.

## Files Description

### `IrisFlower.ipynb`

This Jupyter Notebook file contains the complete code for:

1. Loading and exploring the dataset.
2. Visualizing the data using plots.
3. Splitting the data into training and testing sets.
4. Training and evaluating the Logistic Regression and K-Nearest Neighbors models.
5. Performing hyperparameter tuning using GridSearchCV.
6. Saving the best models.

### `app.py`

A Flask application that provides an API for predicting the iris species using the trained models.

- **Endpoints**:
  - `GET /`: Returns a welcome message.
  - `POST /predict`: Predicts the iris species using both Logistic Regression and K-Nearest Neighbors models. Accepts JSON input with feature values.

### `test_api.py`

A script to test the Flask API by sending a sample request to the `/predict` endpoint and printing the response.

### Data Files

- `iris.zip`: Compressed file containing the dataset.
- `bezdekIris.data`, `iris.data`, `Index`, `iris.names`: Various formats and descriptions of the Iris dataset.

### Model Files

- `knn_model.pkl`: Pickle file containing the trained K-Nearest Neighbors model.
- `log_reg_model.pkl`: Pickle file containing the trained Logistic Regression model.

## Installation and Usage

### Requirements

- Python 3.x
- Required Python packages: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `flask`, `requests`

### Installation

1. Clone the repository or download the project files.
2. Install the required packages using pip:

    ```sh
    pip install pandas matplotlib seaborn scikit-learn joblib flask requests
    ```

3. Unzip the `iris.zip` file to extract the dataset files.

### Running the Jupyter Notebook

1. Open the `IrisFlower.ipynb` file in Jupyter Notebook or JupyterLab.
2. Execute the cells to run the entire analysis and training process.

### Running the Flask API

1. Ensure the trained model files (`log_reg_model.pkl` and `knn_model.pkl`) are in the same directory as `app.py`.
2. Run the Flask application:

    ```sh
    python app.py
    ```

3. The API will be available at `http://127.0.0.1:5000`.

### Testing the API

1. Ensure the Flask server is running.
2. Run the `test_api.py` script to send a test request:

    ```sh
    python test_api.py
    ```

3. You should see the predictions from both models printed in the terminal.

## Example Usage

**Sample request JSON:**

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

