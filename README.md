<div align="center">
  <div>
    <img src="resources/logo.png" width="230" alt="Warp" />
  </div>
    <h1>Coginets</h1>
  <b>
    A Python framework for training, optimizing and deploying Boosting Regressor and Classification models.
  </b><br>
<small>Simplify your fitting and deployment routine today!</small>


<hr />
<img src="https://img.shields.io/github/license/Manuel-Materazzo/Coginets?style=flat-square&logo=opensourceinitiative&logoColor=white&color=gold" alt="license">
<img src="https://img.shields.io/github/last-commit/Manuel-Materazzo/Coginets?style=flat-square&logo=git&logoColor=white&color=gold" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/Manuel-Materazzo/Coginets?style=flat-square&color=gold" alt="repo-top-language">
</div>

## ☄️ Features

|     |                  Feature                   | Description                                                                                                                                                                                 |
|:----|:------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🔄  | **Flexible Training & Validation Options** | Supports multiple methods for training and validating models                                                                                                                                |
| 🛠️ |      **Hyperparameter Optimization**       | Includes built-in hyperparameter tuning with options for **Grid Search**, **Bayesian optimization**, and **Optuna optimization**                                                            |
| 🎻  |            **Model ensembling**            | Provides various methods for combining predictions from multiple models to enhance accuracy, stability, and generalization..                                                                |
| ⚙️  |            **Easy Integration**            | Quickly get started with a new machine learning project by cloning or forking Coginets, you just need to defne a custom pipeline for your dataset.                                          |
| 🌐  |            **Model Deployment**            | Offers model and pipeline serialization and integrates automatically with FastAPI for serving models predictions as an API endpoint.                                                        |
| 🐳  |            **Containerization**            | Includes Dockerfile and Docker Compose configuration for straightforward deployment in any environment.                                                                                     |
| 🤖  |              **CI/CD Ready**               | GitHub Action templates included: <ul><li>🏗️ Automated Docker image build and push</li><li>🔄 Auto-merge requests to sync updates from the base repository</li></ul>                       |
| ✅   |              **Unit Testing**              | Ensures code reliability and correctness through automated unit tests. Adding tests for new modules is straightforward—just compile a configuration class, and they will be auto-generated. |
| 📏  |              **Code Quality**              | Maintains high code quality standards through SonarQube quality gate.                                                                                                                       |

## 🏠 Example project

A complete example is provided in the repository, showcasing Coginets with a home price prediction model. This example
includes:

The dataset for training
A Docker configuration for easy deployment

### 🌐 Live Preview

Check out the live preview on [Render](https://coginets-example.onrender.com/docs) and try out inference.\
Note: The live preview will cold-start as you enter the link, it could take up to 1min to fully load.

## 🚀 Getting Started

### 🐳 Docker prebuilt

1. **Pull the Docker Image**:
   ```sh
   docker pull ghcr.io/manuel-materazzo/coginets-example:latest
    ```
2. **Run the Container**:
   ```sh
   docker run -d -p 8080:80 manuel-materazzo/coginets-example
    ```
3. **Access the API**: Navigate to `http://localhost:8080/docs` or `http://localhost:8080/redoc` to explore the
   interactive API documentation and start making predictions!

### 🐳🔧 Docker compose self-build

1. **Run docker compose**:
   ```sh
   docker-compose up
   ```

### 📦 Manual installation

1. **Clone Coginets repository**:
   ```sh
   git clone https://github.com/your-username/Coginets.git
   cd Coginets
   ```
2. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.