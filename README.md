# IPL Score Prediction

This project uses machine learning models to predict the final score of a batting team in IPL (Indian Premier League) matches based on the first few overs of the innings. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and prediction.

## 📁 Project Structure

```
├── data
│   ├── ipl.csv                  # Raw IPL data
│   └── cleaned.csv              # Cleaned and preprocessed data
│
├── src
│   ├── EDA.ipynb                # Notebook for data processing and model training
│   ├── predictions.ipynb        # Notebook for using the saved model to make predictions
│   ├── model.py                 # Python script for training models
│   └── linear_regression_model.pkl # Saved Linear Regression model
│
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation
```

## 🔍 Overview

- Raw data is cleaned and encoded in `EDA.ipynb`, then used to train and evaluate multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - AdaBoost Regressor

- The best performing model (Linear Regression) is saved for later use.

- `predictions.ipynb` loads the saved model and makes predictions on new inputs.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/08anask/IPL-Score-Prediction.git
   cd IPL-Score-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

- Run `EDA.ipynb` to preprocess the data and train the models.
- Run `predictions.ipynb` to use the saved model to make score predictions.
- Use `model.py` as a standalone script for model training if needed.

## ✅ Dependencies

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  

Install via:
```bash
pip install -r requirements.txt
```

## 🧐 Model Evaluation

The models are evaluated using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

These metrics are printed in the notebook output for comparison.

## 📌 Notes

- Only matches from certain teams are considered for simplification.
- Data is split based on year, with training data up to 2016 and test data from 2017 onward.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
