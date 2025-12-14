# 🛡️ Credit Card Fraud Detection - Full Stack Application

A production-ready, full-stack fraud detection system built with multi-model ensemble machine learning, achieving **97.66% AUC** on real Kaggle data.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📸 Screenshots

### Dashboard Interface
<table>
  <tr>
    <td width="50%">
      <img src="dashboard/assets/pipeline_diagram.png" alt="ML Pipeline Architecture" />
      <p align="center"><b>System Architecture</b></p>
    </td>
    <td width="50%">
      <!-- Add your fraud detection page screenshot here -->
      <p align="center"><b>Fraud Detection Interface</b></p>
      <p align="center"><i>Take a screenshot of the fraud detection page at http://localhost:8501</i></p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <!-- Add model performance screenshot here -->
      <p align="center"><b>Model Performance Dashboard</b></p>
      <p align="center"><i>Screenshot of performance visualizations page</i></p>
    </td>
    <td width="50%">
      <!-- Add admin panel screenshot here -->
      <p align="center"><b>Admin Panel</b></p>
      <p align="center"><i>Screenshot of admin statistics dashboard</i></p>
    </td>
  </tr>
</table>

> **Note**: To add UI screenshots, capture screens from your running application and save them in `dashboard/assets/` directory, then update the image paths above.

## 🎯 Features

### Machine Learning
- **Multi-Model Ensemble**: Combines 4 algorithms (Logistic Regression, Random Forest, Gradient Boosting, Isolation Forest)
- **97.66% AUC Score**: Trained on 284,807 real transactions from Kaggle
- **Real-Time Predictions**: < 100ms response time
- **Comprehensive Metrics**: ROC curves, confusion matrices, precision-recall curves

### Backend (FastAPI)
- **JWT Authentication**: Secure token-based authentication
- **Database Logging**: Every prediction saved to SQLite/PostgreSQL
- **Admin Dashboard API**: System-wide statistics and monitoring
- **RESTful API**: Well-documented endpoints with OpenAPI/Swagger

### Frontend (Streamlit)
- **User Authentication**: Login/Signup with role-based access
- **Fraud Detection Interface**: Submit transactions for analysis
- **Live Monitor**: Real-time transaction simulation
- **Model Performance**: Interactive visualizations
- **Transaction History**: View past predictions
- **Admin Panel**: System statistics (admin-only)

### Infrastructure
- **Docker Support**: One-command deployment
- **Database Flexibility**: SQLite for development, PostgreSQL for production
- **Scalable Architecture**: Microservices-ready design

## 📊 Performance

| Model | AUC Score |
|-------|-----------|
| **Gradient Boosting** 🏆 | **0.9766** |
| Logistic Regression | 0.9660 |
| Random Forest | 0.9621 |
| Ensemble | 0.9668 |
| Isolation Forest | 0.8254 |

Dataset: 284,807 transactions (0.17% fraud rate) from Kaggle

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
# Terminal 1: Start API
python api/app.py

# Terminal 2: Start Dashboard
streamlit run dashboard/app.py
```

4. **Access the application**
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

### Docker Deployment (Recommended)

```bash
docker-compose up -d
```

Access at http://localhost:8501

## 📖 Usage

### Creating an Account

1. Open http://localhost:8501
2. Click "Signup" in the sidebar
3. Enter email and password
4. Start detecting fraud!

### Making Predictions

**Via Dashboard:**
1. Login to the dashboard
2. Go to "🔍 Fraud Detection"
3. Enter transaction details
4. Click "Analyze Transaction"

**Via API:**
```bash
# 1. Signup
curl -X POST "http://localhost:8000/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"yourpassword"}'

# 2. Get your token from response

# 3. Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 100,
    "Amount": 500.00,
    "V1": 0.1, "V2": 0.2, ..., "V28": 0.3
  }'
```

### Admin Access

**Default Admin Credentials:**
- Email: `admin2@fraud.com`
- Password: `admin123`

**Admin Features:**
- System-wide transaction statistics
- Total users and fraud rate
- Pending fraud alerts
- Blocked transactions

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Frontend             │
│  - Authentication UI                │
│  - Fraud Detection Interface        │
│  - Admin Dashboard                  │
└──────────────┬──────────────────────┘
               │ HTTP/REST
┌──────────────▼──────────────────────┐
│      FastAPI Backend                │
│  - JWT Auth Middleware              │
│  - ML Model Inference               │
│  - Database ORM                     │
└──────────────┬──────────────────────┘
               │ SQLAlchemy
┌──────────────▼──────────────────────┐
│   PostgreSQL / SQLite Database      │
│  - Users & Authentication           │
│  - Transactions & Predictions       │
│  - Fraud Alerts                     │
└─────────────────────────────────────┘
```

## 📁 Project Structure

```
credit_card_fraud_detection/
├── api/
│   └── app.py              # FastAPI application
├── auth/
│   └── auth.py             # JWT authentication
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── assets/             # Visualization plots
├── database/
│   ├── db.py               # SQLAlchemy models
│   ├── crud.py             # Database operations
│   └── schema.sql          # PostgreSQL schema
├── data/
│   ├── creditcard.csv      # Kaggle dataset
│   └── processed/          # Preprocessed data
├── models/
│   ├── *_model.pkl         # Trained models
│   └── ensemble_config.json
├── src/
│   ├── preprocessing.py    # Data preprocessing
│   └── train_ensemble.py   # Model training
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── requirements.txt
└── README.md
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# Database
DATABASE_URL=sqlite:///./fraud_detection.db
# For PostgreSQL: DATABASE_URL=postgresql://user:pass@localhost:5432/fraud_db

# JWT
JWT_SECRET_KEY=your-super-secret-key-change-this-in-production

# API
API_URL=http://localhost:8000
```

## 🧪 Model Training

To retrain models on your own data:

```bash
# 1. Preprocess data
python src/preprocessing.py

# 2. Train ensemble
python src/train_ensemble.py
```

This will generate:
- Trained models in `models/`
- Performance visualizations in `dashboard/assets/`
- Metrics in `models/ensemble_metrics.json`

## 📊 API Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/` | GET | API info | No |
| `/auth/signup` | POST | Create account | No |
| `/auth/login` | POST | Get JWT token | No |
| `/auth/me` | GET | User info | Yes |
| `/predict` | POST | Fraud prediction | Yes |
| `/transactions` | GET | User history | Yes |
| `/admin/stats` | GET | System stats | Admin |
| `/admin/alerts` | GET | Fraud alerts | Admin |

## 🔐 Security

- **Password Hashing**: bcrypt with salt
- **JWT Tokens**: 24-hour expiration
- **Role-Based Access**: User/Admin roles
- **Database Isolation**: Users see only their data
- **SQL Injection Prevention**: SQLAlchemy ORM
- **HTTPS Ready**: Configure reverse proxy

## 🐳 Docker Deployment

**Services:**
- `postgres`: PostgreSQL database
- `api`: FastAPI backend
- `dashboard`: Streamlit frontend

**Commands:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild
docker-compose up -d --build
```

## 📈 Performance Optimization

- **Model Caching**: Models loaded once at startup
- **Database Indexing**: Optimized queries
- **Connection Pooling**: SQLAlchemy engine
- **Async Ready**: FastAPI async support
- **Batch Predictions**: API supports bulk requests

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Web Frameworks**: FastAPI, Streamlit
- **Database**: PostgreSQL, SQLAlchemy

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/credit-card-fraud-detection](https://github.com/yourusername/credit-card-fraud-detection)

---

**Built with ❤️ for fraud detection**
