# ChurnSight AI 📊

**ChurnSight AI** is a comprehensive, AI-powered customer retention platform. It utilizes ensemble machine learning models to predict customer churn, estimates Customer Lifetime Value (CLV), and provides actionable insights through interactive "What-If" simulations. Furthermore, the platform features an automated email outreach system to help Customer Success teams proactively engage at-risk customers.

![ChurnSight AI Dashboard Preview](https://via.placeholder.com/800x400?text=ChurnSight+AI+Dashboard)

## 🌟 Key Features

1. **High-Accuracy Churn Prediction**
   - Utilizes an advanced ensemble model (CatBoost, LightGBM, and Deep Neural Networks).
   - Identifies the top factors driving a customer's decision to churn (e.g., Month-to-month contracts, lack of Tech Support).

2. **Customer Lifetime Value (CLV) Estimation**
   - Predicts the future revenue potential of each customer based on their tenure, contract type, and monthly charges.
   - Segments customers into value tiers (Standard, High, VIP) to prioritize retention efforts.

3. **Interactive "What-If" Simulator**
   - Allows Customer Success managers to simulate changes to a customer's plan (e.g., upgrading to a 1-year contract, adding Tech Support).
   - Instantly recalculates the new churn probability to show the impact of the proposed changes.

4. **Automated Retention Email Outreach**
   - Generates highly personalized, corporate-grade email templates based on the specific reasons a customer is at risk (e.g., Price Sensitivity vs. Support needs).
   - Integrated backend SMTP service to send emails directly from the dashboard to the customer.

## 💻 Tech Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (ES6+), Chart.js for data visualization.
- **Backend**: FastAPI (Python) for high-performance API endpoints.
- **Machine Learning**: `scikit-learn`, `CatBoost`, `LightGBM`, `PyTorch`.
- **Deployment Ready**: Configured for serverless deployment on Vercel (`vercel.json`).

## 🚀 Getting Started (Local Development)

Follow these instructions to run the ChurnSight AI platform locally.

### Prerequisites
- Python 3.9+ 
- Node.js (Optional, if using npm scripts)
- A Gmail account with an "App Password" generated for the email sending feature.

### 1. Clone the Repository
```bash
git clone https://github.com/SakthiV4/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Backend Dependencies
Create a virtual environment and install the required Python packages:

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory of the project and add your Gmail credentials for the email service:

```env
SENDER_EMAIL=your_email@gmail.com
MAIL_APP_PASSWORD=your_16_character_app_password
```
*(Note: Never commit your `.env` file to version control. It is already included in `.gitignore`.)*

### 4. Run the Application
You can start the backend FastAPI server using:

```bash
uvicorn api.predict:app --reload --port 8000
```
*(Or if you have the npm scripts set up, simply run `npm run dev`)*

Next, open the `frontend/index.html` file in your web browser (or use an extension like VS Code Live Server) to view the dashboard.

## 📁 Repository Structure

- `/api` - Contains the FastAPI backend application (`predict.py`).
- `/frontend` - Contains the HTML, CSS, and JS files for the dashboard user interface.
- `/src` & `/notebooks` - Scripts and Jupyter notebooks used for data preprocessing and training the machine learning models.
- `/models` - Serialized `.pkl` and `.pth` files containing the trained predictive models.
- `/data` - The Telco Customer Churn dataset used for training the model.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
This project is licensed under the MIT License.
