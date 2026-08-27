# mlops-pytorch-pipeline

A production-style MLOps pipeline for training and serving a PyTorch image classification model (CIFAR-10) using Docker and Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                        │
│  main ← develop ← feature/docker-training                      │
│                  ← feature/k8s-deployment                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ CI (GitHub Actions)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Images                            │
│  mlops-train:v1  (multi-stage, PyTorch + training deps)        │
│  mlops-serve:v1  (slim, FastAPI + inference deps only)         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│           Kubernetes Namespace: ml-training                     │
│                                                                 │
│  ConfigMap ──► Training Job ──► PVC (checkpoints)              │
│                                        │                        │
│                                        ▼                        │
│              Serving Deployment (2 replicas) ──► Service       │
│              + HPA (auto-scaling)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── train.py        # Training loop with early stopping
│   ├── model.py        # ResNet-18 CNN model
│   ├── dataset.py      # CIFAR-10 data loading
│   └── serve.py        # FastAPI serving app
├── configs/
│   └── training_config.yaml
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt
│   └── serve.txt
└── tests/
    └── test_model.py
```

## Prerequisites

- Python 3.10+
- Docker Desktop
- `kubectl` CLI
- A Kubernetes cluster (Minikube, kind, or cloud-managed)
- A GitHub account

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/mlops-pytorch-pipeline.git
cd mlops-pytorch-pipeline
git checkout -b develop
```

### 2. Local Python Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements/train.txt
```

### 3. Run Training Locally

```bash
python src/train.py
```

### 4. Run Serving Locally

```bash
python src/serve.py
# Test: curl -X GET http://localhost:8080/health
```

## Docker Usage

### Build & Run Training Image

```bash
# Build
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

### Build & Run Serving Image

```bash
# Build
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## Kubernetes Deployment

### Apply All Manifests

```bash
# Create namespace and config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Run training job
kubectl apply -f k8s/training-job.yaml

# Monitor training
kubectl get pods -n ml-training
kubectl logs -f job/pytorch-training -n ml-training

# Deploy serving layer (after training completes)
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### Verify Deployment

```bash
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

### Test Prediction Endpoint

```bash
# Port-forward for local testing
kubectl port-forward svc/model-serving 8080:80 -n ml-training

# Send a prediction request
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns 200 if model is loaded |
| `/predict` | POST | Accepts image, returns class probabilities |

### Example Response

```json
{
  "predictions": [
    {"class": "airplane", "probability": 0.85},
    {"class": "automobile", "probability": 0.07},
    ...
  ],
  "predicted_class": "airplane"
}
```

## Git Workflow

- All work is done on feature branches
- Feature branches are merged to `develop` via Pull Requests
- `develop` is merged to `main` for releases
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)

## CIFAR-10 Classes

| Index | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |
