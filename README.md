# RetentionLens AI - Customer Churn Prediction & Action Platform

## Overview
RetentionLens AI is an advanced, end-to-end Machine Learning web application designed to predict customer churn, calculate Customer Lifetime Value (CLV), and generate actionable retention strategies. It provides a robust API powered by FastAPI, an intuitive web interface, and an automated email retention system.

## Key Features
- **Accurate Churn Prediction**: Utilizes a highly optimized CatBoost/Scikit-learn pipeline to predict the probability of a customer leaving.
- **Feature Impact Analysis**: Explains *why* a customer is at risk by analyzing key factors (e.g., contract type, monthly charges, tenure).
- **AI Strategy Generator**: Dynamically generates tailored retention strategies mimicking an LLM, providing actionable scripts for customer support.
- **Customer Lifetime Value (CLV)**: Predicts the expected revenue from a customer, segmenting them into VIP, High Value, Standard, and Low Value tiers.
- **Batch Processing**: Supports uploading CSV files to evaluate up to 5,000 customers at once, identifying high-risk segments efficiently.
- **Automated Email Outreach**: Integrates with Gmail SMTP to send retention emails directly from the dashboard.
- **Interactive Dashboard**: A responsive, modern frontend built with HTML/CSS/JS that communicates seamlessly with the FastAPI backend.

## Architecture & Tech Stack
- **Backend Framework**: FastAPI (Python)
- **Machine Learning**: Scikit-Learn, CatBoost, Pandas, Joblib
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Deployment**: Configured for deployment on Vercel (`vercel.json` included)
- **Security & Rate Limiting**: SlowAPI integration to prevent API abuse.

## Directory Structure
- `api/`: Contains the core FastAPI application (`predict.py`) and endpoints.
- `src/`: Data preprocessing and model training scripts (`train.py`, `train_clv.py`, `preprocess.py`).
- `frontend/`: Static HTML/JS/CSS files for the user interface (`index.html`, `batch.html`).
- `models/`: Pickled ML pipelines (`churn_pipeline.pkl`, `clv_pipeline.pkl`) and feature configurations.
- `data/`: Dataset storage for training and testing.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model prototyping.

## Installation & Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SakthiV4/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   FRONTEND_URL=http://localhost:8000
   SENDER_EMAIL=your-email@gmail.com
   MAIL_APP_PASSWORD=your-gmail-app-password
   ```

4. **Run the Application:**
   ```bash
   uvicorn api.predict:app --reload --port 8000
   ```
   Access the frontend at `http://localhost:8000/`.

## API Endpoints
- `GET /api/health`: Health check endpoint.
- `POST /api/predict`: Returns churn probability, risk level, CLV, feature impacts, and retention strategy for a single customer.
- `POST /api/predict/batch`: Processes a CSV file and returns a sorted list of high-risk customers.
- `POST /api/send-email`: Sends a retention email (rate-limited).

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
