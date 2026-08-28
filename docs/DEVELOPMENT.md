# ModelForge AI — Local Developer Setup Guide

## Prerequisites
- **Python 3.11+ / 3.12+**
- **Node.js 18+ / 20+** and **npm**
- **Docker & Docker Compose** (Optional for containerized run)
- **Git**

---

## 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend development server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be live at `http://localhost:8000`. OpenAPI docs: `http://localhost:8000/api/v1/docs`.

---

## 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Start Next.js development server
npm run dev
```
Frontend Web Dashboard will be live at `http://localhost:3000`.

---

## 3. Seeding Sample Enterprise Projects

```bash
python scripts/seed_data.py
```
This populates the database with realistic sample datasets (Fraud Detection), experiments, XGBoost / LightGBM models, Canary deployments, and drift simulations.
