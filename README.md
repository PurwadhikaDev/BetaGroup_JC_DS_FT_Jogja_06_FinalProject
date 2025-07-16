# Olist Brazilian E-Commerce Analysis

## Project Overview

This project analyzes the Brazilian E-Commerce Public Dataset by Olist, a marketplace platform connecting small and medium businesses to major e-commerce channels in Brazil. The dataset, spanning 2016 to 2018, includes over 100,000 orders with details on customer orders, payments, reviews, order items, sellers, products, and geolocation data. The goal is to address key operational and strategic challenges faced by Olist, focusing on logistics optimization, customer satisfaction, revenue growth, product prioritization, payment method diversification, regional market expansion, and predictive modeling for customer satisfaction.

The analysis employs exploratory data analysis (EDA), data preprocessing, and machine learning to develop a predictive model using XGBoost with SMOTEENN resampling and feature importance-based selection to predict customer satisfaction and identify at-risk customers. The project delivers actionable insights to enhance operational efficiency, improve customer experiences, and drive sustainable growth.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Business Problem](#business-problem)
- [Project Goals](#project-goals)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [Recommendations](#recommendations)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributors](#contributors)
- [License](#license)

## Dataset

The dataset is sourced from [Kaggle: Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It includes:
- **Orders**: Order IDs, customer IDs, timestamps, and statuses.
- **Payments**: Payment values, methods, and installments.
- **Reviews**: Review scores, comments, and timestamps.
- **Order Items**: Product IDs, quantities, prices, and freight costs.
- **Products**: Categories, dimensions, and weights.
- **Sellers and Customers**: Geolocation and demographic data.

The dataset contains 44 features and supports comprehensive analysis of marketplace dynamics, customer behavior, and logistics.

## Business Problem

Olist operates in a complex e-commerce landscape with challenges including:
1. **Logistics & Freight Costs**: High freight costs and delivery inefficiencies due to Brazil’s vast geography.
2. **Customer Satisfaction**: Identifying drivers of dissatisfaction to reduce churn and improve experience.
3. **Revenue Decline**: Addressing declining payment values through pricing and promotional strategies.
4. **Product Optimization**: Prioritizing high-performing product categories to maximize revenue.
5. **Payment Diversification**: Expanding beyond credit cards to include modern methods like Pix.
6. **Regional Expansion**: Targeting underserved regions to leverage Brazil’s 15.7% e-commerce CAGR through 2030.
7. **Predictive Modeling**: Building a model to predict customer satisfaction and detect churn risk.

## Project Goals

- Optimize logistics to reduce freight costs while maintaining on-time delivery.
- Enhance customer satisfaction by addressing delivery, product, and category issues.
- Reverse revenue decline through targeted pricing and promotion strategies.
- Prioritize high-performing product categories to improve resource allocation.
- Diversify payment methods to improve checkout efficiency and customer reach.
- Expand into high-potential underserved regions to grow market share.
- Develop a predictive machine learning model to identify factors affecting customer satisfaction and churn risk.

## Methodology

1. **Data Loading and Preprocessing**:
   - Merged multiple datasets (orders, payments, reviews, etc.) into a unified DataFrame.
   - Handled missing values, outliers, and categorical encoding.
   - Applied feature engineering (e.g., delivery time, review response time).

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed customer satisfaction trends, freight costs, payment methods, and regional distribution.
   - Identified correlations between features like delivery time and review scores.

3. **Machine Learning**:
   - Built a classification model using XGBoost with SMOTEENN resampling to address class imbalance.
   - Performed hyperparameter tuning with RandomSearchCV.
   - Applied feature importance-based selection to streamline the model.

4. **Model Evaluation**:
   - Evaluated using weighted F2-score, precision, recall, and accuracy.
   - Analyzed confusion matrix to assess performance on satisfied vs. unsatisfied customers.

5. **Deployment**:
   - Saved the trained model using `joblib` for future predictions.
   - Demonstrated a simple prediction pipeline for new data.

## Key Findings

- **Logistics**: Freight costs significantly impact total payment values, especially for heavy products.
- **Customer Satisfaction**: Delivery delays and product quality are key drivers of dissatisfaction.
- **Revenue Trends**: Declining payment values correlate with reduced order volumes in certain categories.
- **Product Categories**: High-performing categories (e.g., electronics, housewares) drive revenue but vary in freight costs.
- **Payment Methods**: Credit cards dominate, but emerging methods like Pix show growth potential.
- **Regional Insights**: Underserved regions in the North and Northeast present expansion opportunities.

## Model Performance

The XGBoost model with SMOTEENN resampling achieved:
- **Test F2-Score**: 0.8013
- **Test Accuracy**: 82.48%
- **Test Precision**: 83.38%
- **Test Recall**: 82.48%

**Confusion Matrix**:
- True Negatives (Unsatisfied): 778
- False Positives (Unsatisfied as Satisfied): 3,178
- True Positives (Satisfied): 14,700
- False Negatives (Satisfied as Unsatisfied): 109

The model excels at predicting satisfied customers but struggles with identifying unsatisfied ones, indicating a bias towards the majority class.

## Limitations

- **Minority Class Performance**: High false positives for unsatisfied customers suggest bias towards the majority class.
- **Synthetic Data Risks**: SMOTEENN may introduce overfitting if synthetic samples do not represent real-world patterns.
- **Computational Complexity**: XGBoost with hyperparameter tuning is resource-intensive.
- **Interpretability**: The model’s black-box nature limits straightforward business rule extraction.
- **Precision-Recall Trade-off**: Further tuning is needed to prioritize recall for unsatisfied customers if critical to business goals.

## Recommendations

1. **Threshold Adjustment**: Adjust the decision threshold to improve recall for unsatisfied customers.
2. **Alternative Resampling**: Explore SMOTETomek or cost-sensitive learning to enhance minority class detection.
3. **Explainability Tools**: Use SHAP or LIME to improve model interpretability for stakeholders.
4. **Synthetic Data Validation**: Monitor SMOTEENN-generated samples for real-world representativeness.
5. **Business Alignment**: Prioritize recall for unsatisfied customers if misclassification risks are high.
6. **Future Iterations**: Test simpler models (e.g., logistic regression) or ensemble approaches for improved performance.

## Installation

To run this project, ensure you have Python 3.11+ installed. Install the required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost imbalanced-learn joblib
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/olist-ecommerce-analysis.git
   cd olist-ecommerce-analysis
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place it in the project directory.

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook JCDS060_JOG_Team_Beta_RandomSearch_Brazilian_ecommerce_ML_Final.ipynb
   ```

4. To make predictions with the saved model:
   ```python
   import joblib
   import pandas as pd

   # Load the model
   model = joblib.load('best_model.pkl')

   # Example dummy data
   df_dummy = pd.DataFrame([{
       'review_time_days': -2,
       'processing_time_days': 10,
       'quantity': 6,
       'payment_installments': 8,
       'review_response_time_days': 5,
       'delivery_time_days': 20
   }])

   # Predict
   prediction = model.predict(df_dummy)
   print("Satisfied" if prediction[0] == 1 else "Not Satisfied")
   ```

## File Structure

```
olist-ecommerce-analysis/
├── JCDS060_JOG_Team_Beta_RandomSearch_Brazilian_ecommerce_ML_Final.ipynb
├── best_model.pkl
├── README.md
├── data/ (place Kaggle dataset here)
└── requirements.txt
```

## Contributors

- **Ade Ilma Hasanah**
- **Asiyatul Mahfudloh**
- **Frans Julian Bryan Sagala**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.