# 🚀 Real-Time Fraud Transaction Detection Pipeline

This repository contains the code for an end-to-end machine learning project designed to detect fraudulent financial transactions in real-time. The project progresses from in-depth exploratory data analysis and advanced feature engineering to hyperparameter tuning of an XGBoost model and its deployment as an interactive web application.

The final model is deployed using Streamlit and can be accessed via the live demo link below.

---
### 🚀 **Live Demo**

You can interact with the live web application here:
**[Live Fraud Detection Dashboard](https://fraud-transaction-detection-pipeline-bg33chnvzh5ybpu9b44ybl.streamlit.app/)**



---
### 🛠️ **Tech Stack & Libraries**

| Technology | Description |
| :--- | :--- |
| **Python** | Core language for data analysis and modeling. |
| **Pandas & NumPy** | Used for efficient data manipulation and numerical operations. |
| **Scikit-Learn** | Utilized for data preprocessing, model training, and evaluation. |
| **XGBoost** | The primary gradient boosting library for building the high-performance model. |
| **Matplotlib & Seaborn** | Employed for data visualization during Exploratory Data Analysis (EDA). |
| **SHAP** | Used for model interpretability and explaining feature importance. |
| **Streamlit** | The framework for building and deploying the interactive web dashboard. |
| **Joblib/Pickle** | Used for saving and loading the trained model and other necessary objects. |

---
### 📂 **Project Structure**

| File/Folder | Description |
| :--- | :--- |
| `app.py` | The main script for the Streamlit web application. |
| `prediction.py` | Contains the core logic for loading the model and making predictions. |
| `xgb_model.pkl` | The final, tuned XGBoost model saved after training. |
| `requirements.txt` | Lists all project dependencies required to run the application. |
| `Fraud_Transaction_Detection.ipynb` | The Jupyter notebook containing the full workflow from EDA to model training. |
| `assets/` | A folder containing the background image and performance plots for the dashboard. |

---
### 📈 **Model Performance**

After extensive feature engineering and hyperparameter tuning, the final **XGBoost** model demonstrated excellent performance on the unseen test set:

-   ✅ **ROC-AUC Score:** **0.9995**, indicating an outstanding ability to distinguish between fraudulent and legitimate transactions.
-   ✅ **Precision-Recall AUC:** **0.9445**, confirming strong performance on the highly imbalanced dataset.
-   At the optimal threshold of **0.97**, the model achieved a high **F1-Score of 0.89**, successfully balancing the need to catch fraud (Recall) while minimizing false alarms (Precision).

---
### 🧠 **Model Interpretability with SHAP**

To ensure the model's decisions are transparent, I used **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions.

#### 🔑 **Key Insights:**
-   **Recent Fraud History:** The most powerful predictors were features tracking recent fraudulent activity on a customer's account (`customer_fraud_rolling_7d`) and at a specific terminal (`terminal_fraud_28d`).
-   **Transaction Amount:** The `TX_AMOUNT` was also a major contributor, with higher values strongly indicating a higher risk of fraud.
-   These insights confirm that the model learns patterns that align with real-world fraud scenarios, increasing trust in its predictions.

---
### 🔧 Running the Dashboard Locally

To get this application running on your own machine, please follow the steps below to set up the environment and launch the server.

* **1. Get a Local Copy**

    First, you'll need a local copy of the project. Open your terminal, navigate to your desired directory, and clone the repository using `git`.

    ```bash
    git clone https://github.com/your-username/Fraud-Transaction-Detection-Pipeline.git
    cd Fraud-Transaction-Detection-Pipeline
    ```

* **2. Set Up the Environment**

    It is highly recommended to use a virtual environment to keep the project's dependencies isolated. The following commands will create a new environment and activate it.

    ```bash
    # Create the environment
      python -m venv .venv

    # Activate the environment (use the command for your OS)
    # Windows:
    .\.venv\Scripts\activate
    
    # macOS/Linux:
    # source .venv/bin/activate
    ```

* **3. Install Dependencies**

    With the environment activated, install all the required libraries from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

* **4. Launch the Application**

    Finally, start the Streamlit web server. Once the server is running, it will automatically open the dashboard in your default web browser.

    ```bash
    streamlit run app.py
    ```

    After running the command, your web browser should open the application automatically. If it doesn't, open a new tab and navigate to http:// localhost:8501 to view the dashboard.
  
🙌 Closing Thoughts:

This project demonstrates a comprehensive approach to solving a critical financial problem using machine learning. By building a highly accurate and interpretable model, this pipeline showcases the potential of AI to create robust systems that can enhance security and build trust in digital transactions. While not a production-ready system, it serves as a strong proof-of-concept for data-driven fraud detection.

---
### 👤 **Author**

- **Swaransh Mishra**
- **GitHub:** [@Swaransh-Mishra](https://github.com/Swaransh-Mishra)
- **LinkedIn:** [Swaransh Mishra](https://www.linkedin.com/in/#)
