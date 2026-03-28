# Adult

> Predict income levels based on demographic data using machine learning.

## 🧠 Overview

This project focuses on predicting whether an individual's income exceeds $50K per year using the Adult dataset, which contains various demographic features. The core problem is to build a robust machine learning model that can accurately classify income brackets, addressing challenges such as class imbalance and feature engineering.

## 🔨 What I Built

I built a machine learning pipeline that includes comprehensive Exploratory Data Analysis (EDA) and the implementation of several classification models. The project tackles the income prediction problem by:

- **Data Preprocessing:** Cleaning categorical features by removing whitespace, handling missing values, and identifying redundant or high-cardinality features for removal.
- **Feature Engineering:** Imputing missing values and preparing numerical and categorical features for modeling.
- **Class Imbalance Handling:** Employing both undersampling (RandomUnderSampler) and oversampling (RandomOverSampler) techniques to balance the target variable classes.
- **Model Selection & Training:** Implementing and evaluating Decision Trees, Random Forests, and Gradient Boosting Classifiers.
- **Hyperparameter Tuning:** Utilizing GridSearchCV to optimize the performance of the chosen models.
- **Interactive EDA:** Developing interactive visualization functions in `eda.py` using `ipywidgets`, `matplotlib`, and `seaborn` to dynamically explore relationships between numerical features and the target variable.

## 💭 Thought Process

My approach to this problem began with a thorough Exploratory Data Analysis (EDA) in `Adult.ipynb` to understand the dataset's characteristics, including missing values, cardinality, and class distribution. I decided to drop features like 'education', 'fnlgwt', 'capital_gain', and 'capital_loss' due to redundancy or a high number of zero values, which I found to be less informative during the initial analysis.

A significant challenge was the class imbalance in the target variable (`income >50K` vs `<=50K`). To address this, I chose to experiment with both undersampling and oversampling techniques to see their impact on model performance. I also decided to implement different classification algorithms (Decision Trees, Random Forests, and Gradient Boosting Classifiers) to compare their effectiveness on this dataset. The use of `GridSearchCV` was crucial for systematically finding the best hyperparameters for each model.

In parallel, I developed `eda.py` to create interactive visualization tools. This was a key decision to enhance the EDA process, allowing for dynamic exploration of feature distributions and their correlation with the target. This interactive approach proved invaluable in quickly identifying patterns and gaining insights, rather than relying on static plots.

Throughout the process, I learned the importance of iterative refinement in feature selection and the significant impact of handling class imbalance on model accuracy, precision, and recall. The modularity of separating EDA functions into `eda.py` also proved beneficial for reusability and cleaner code within the notebook.

## 🛠️ Tools & Tech Stack

| Layer      | Technology               |
|------------|--------------------------|
| Language   | Python                   |
| Libraries  | pandas                   |
|            | matplotlib               |
|            | seaborn                  |
|            | plotly                   |
|            | scikit-learn             |
|            | imblearn                 |
|            | category_encoders        |
| Environment| Jupyter Notebook         |

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required Python libraries (listed below)

### Installation

```bash
git clone https://github.com/rashadmin/Adult.git
cd Adult
pip install -r requirements.txt # Assuming a requirements.txt exists based on tools_or_frameworks_used
```

_Note: A `requirements.txt` file is not explicitly provided in the summaries, but it is a standard practice for Python projects. If one does not exist, you can create it with the following content:_

```
pandas
matplotlib
seaborn
plotly
scikit-learn
imbalanced-learn
category_encoders
ipywidgets
```

### Environment Variables

This project does not require any specific environment variables.

### Run

To run the main analysis and modeling:

```bash
jupyter notebook Adult.ipynb
```

Open `Adult.ipynb` in your browser and run all cells.

## 📖 Usage

### Example 1: Interactive EDA (from `eda.py`)

The `eda.py` file provides functions to create interactive plots within a Jupyter Notebook. For example, to visualize numerical features against the target:

```python
from eda import plot_numerical_features_interactive
from ipywidgets import interact
import pandas as pd

# Load your dataframe (replace with your data loading)
df = pd.read_csv('adult.data', skipinitialspace=True)
df.columns = ['age', 'workclass', 'fnlgwt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
df['target'] = (df['income'] == '>50K').astype(int) # Example target creation

numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
if 'target' in numerical_cols:
    numerical_cols.remove('target')

interact(plot_numerical_features_interactive, df=fixed(df), features=fixed(numerical_cols));
```

### Example 2: Running the Income Prediction Model

The `Adult.ipynb` notebook demonstrates the full pipeline. After loading the data, performing EDA, and preprocessing, the models are trained and evaluated. The final cells will display model performance metrics.

## 📚 Resources

- [pandas Documentation](https://pandas.pydata.org/docs/) — Data manipulation and analysis
- [Matplotlib Documentation](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) — Plotting library
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html) — Statistical data visualization
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/) — High-level interface for Plotly
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) — Machine learning library
- [Imbalanced-learn Documentation](https://imbalanced-learn.google.com/stable/) — Library for imbalanced datasets
- [Category Encoders Documentation](https://contrib.scikit-learn.org/category_encoders/) — Categorical feature encoders
- [ipywidgets Documentation](https://ipywidgets.readthedocs.io/en/latest/examples/Using%20interact.html) — Interactive HTML widgets
- [Exploratory Data Analysis (EDA) Concepts](https://www.coursera.org/lecture/data-analysis-with-python/exploratory-data-analysis-8t31j) — Overview of EDA
- [Class Imbalance Concepts](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-a-guide-for-data-scientists-e0e645a278d6) — Understanding and addressing imbalanced datasets
- [Feature Engineering Concepts](https://machinelearningmastery.net/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/) — Creating new features
- [Model Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) — Accuracy, Recall, Precision
- [Cross-Validation Concepts](https://scikit-learn.org/stable/modules/cross_validation.html) — Model validation technique
- [Hyperparameter Tuning Concepts](https://scikit-learn.org/stable/modules/grid_search.html) — Optimizing model parameters

## 📄 License

MIT © [rashadmin](https://github.com/rashadmin)
