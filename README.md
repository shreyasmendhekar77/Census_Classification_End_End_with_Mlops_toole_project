# 📦 End-to-End Machine Learning Project with MLflow & AWS CI/CD Deployment

This project demonstrates a complete machine learning workflow — from model development to experiment tracking with **MLflow**, and automated deployment using **Docker, GitHub Actions**, and **AWS (ECR + EC2)**.

---

## 🚀 Project Workflow

1. Update `config.yaml`
2. Update `schema.yaml`
3. Update `params.yaml`
4. Update the Entity module
5. Update Configuration Manager in `src/config`
6. Build components (data ingestion, model, etc.)
7. Create pipelines to connect components
8. Implement `main.py` (pipeline entry point)
9. Build a Flask API in `app.py` for deployment

---

## 💠 How to Run Locally

### 📁 Step 1: Clone the repository

```bash
git clone https://github.com/entbappy/End-to-end-Machine-Learning-Project-with-MLflow.git
cd End-to-end-Machine-Learning-Project-with-MLflow
```

### 🐍 Step 2: Create and activate conda environment

```bash
conda create -n mlproj python=3.8 -y
conda activate mlproj
```

### 📦 Step 3: Install required packages

```bash
pip install -r requirements.txt
```

### ▶️ Step 4: Run the app

```bash
python app.py
```

Now open `http://localhost:5000` in your browser.

---

## 📊 Experiment Tracking with MLflow

MLflow is used to log experiments, parameters, metrics, and models.

### 📚 MLflow Docs

[https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

### ▶️ Run MLflow UI locally

```bash
mlflow ui
```

Then open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### ☁️ Use MLflow with DagsHub

If using DagsHub as MLflow backend:

```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
set MLFLOW_TRACKING_USERNAME=<your-username>
set MLFLOW_TRACKING_PASSWORD=<your-password>
```

Replace with your actual credentials and repo.

---

## 🐳 Docker + AWS CI/CD with GitHub Actions

### ✅ Prerequisites

* AWS account
* IAM user with permissions:

  * `AmazonEC2FullAccess`
  * `AmazonEC2ContainerRegistryFullAccess`
* GitHub repository
* EC2 instance (Ubuntu)

---

### 🌐 Steps to Deploy

#### 🛠️ 1. Create ECR Repository

Save the URI, e.g.:

```bash
507319107220.dkr.ecr.us-west-1.amazonaws.com/mlops_project
```

#### 🔦 2. Launch EC2 Instance

Ubuntu recommended.

#### 🐳 3. Install Docker in EC2

```bash
sudo apt update && sudo apt upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

#### 🤖 4. Configure EC2 as GitHub Self-Hosted Runner

* Go to your repo > Settings > Actions > Runners > Add runner
* Choose OS (Ubuntu) and run the setup commands on EC2.

#### 🔐 5. Set GitHub Secrets

| Name                    | Value                                                |
| ----------------------- | ---------------------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | Your AWS access key                                  |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key                                  |
| `AWS_REGION`            | e.g., `us-west-1`                                    |
| `AWS_ECR_LOGIN_URI`     | e.g., `507319107220.dkr.ecr.us-west-1.amazonaws.com` |
| `ECR_REPOSITORY_NAME`   | e.g., `mlops_project`                                |

---

## ⚙️ What GitHub Actions Does

* Build Docker image of your ML app
* Push image to Amazon ECR
* SSH into EC2 instance
* Pull image and run container with your Flask app

---

## ✅ Key Features

* Full ML project pipeline with modular code
* MLflow integration for tracking experiments
* Dockerized application
* CI/CD using GitHub Actions and AWS (ECR + EC2)
* Scalable and production-ready architecture

---

## 💡 About MLflow

MLflow enables:

* Logging and comparing experiments
* Registering and serving models
* Production-grade deployment support

---

## 📬 Contact

**Shreyas Mendhekar**
[GitHub](https://github.com/shreyasmendhekar77)
[LinkedIn](https://www.linkedin.com/in/shreyasmendhekar/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
