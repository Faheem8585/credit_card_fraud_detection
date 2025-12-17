# Railway Deployment
This project can be deployed to Railway for the backend API.

## Quick Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/Faheem8585/credit_card_fraud_detection)

## Manual Deployment

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Choose `Faheem8585/credit_card_fraud_detection`
5. Configure:
   - **Start Command**: `uvicorn api.app:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `DATABASE_URL`: (Railway will auto-generate if you add PostgreSQL)
     - `JWT_SECRET_KEY`: `your-secret-key-change-this`

6. Get your API URL (e.g., `https://fraud-api.railway.app`)

## Update Streamlit Cloud

After deploying the backend:

1. Go to Streamlit Cloud dashboard
2. Click your app → Settings → Secrets
3. Add:
   ```toml
   API_URL = "https://your-backend-url.railway.app"
   ```

4. Save and reboot the app
