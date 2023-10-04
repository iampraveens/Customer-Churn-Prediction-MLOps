# Customer Churn Prediction <img src="https://cdn-icons-png.flaticon.com/512/9815/9815472.png" alt="Car Price Prediction" width="50" height="50">

Telcom Churn Prediction is a machine learning project that predicts whether a customer is likely to stay with a telecom service provider or churn. This repository contains the code and resources for the predictive model, along with a Streamlit web application for easy user interaction.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Code Layout](#code-layout)
- [Getting Started](#getting-started)
- [Training the Model](#training-the-model)
- [Save the Model](#save-the-model)
- [Prediction](#prediction)
- [Dockerized Web App](#dockerized-web-app)
- [Experimental Tracking](#experimental-tracking)
- [License](#license)

## Overview
Telecom Churn Prediction is a machine learning project that predicts whether a customer is likely to stay with a telecom service provider or churn (cancel their subscription). It leverages data preprocessing techniques and employs an RandomForest classification model for accurate predictions. The project is organized into several modules for easy management and scalability.

## Project Structure
The project structure is organized as follows:

- `data/`: Contains the dataset (`car_data.csv`) used for training and predictions.
- `pipelines/`: Includes ZenML pipelines for data cleaning, model training, and evaluation.
- `steps/`: Custom Python scripts for data loading, model training, and evaluation.
- `src/`: Source code files, including data cleaning strategies, model development, and utilities.
- `saved_models/`: Stores trained machine learning models.
- `utils.py`: Utility functions for model saving and loading.
- `app.py`: Streamlit-based web application for predicting car prices.
- `requirements.txt`: Python dependencies for the project.
- `Dockerfile`: Docker configuration for containerizing the web app.

## Code Layout
![Customer Churn_layout](https://github.com/iampraveens/Customer-Churn-Prediction-MLOps/assets/125688218/2916cfa4-f922-40a1-b693-661ed21eb7fd)

## Getting Started
To get started with the project, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/iampraveens/Customer-Churn-Prediction-MLOps.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```
## Training the Model

```bash
python ./steps/run_pipeline.py
```
- This command will execute the data cleaning, model training, and evaluation process

## Save the Model

```bash
python ./steps/save_model.py
```

## Prediction

```bash
streamlit run app.py
```
## Experimental Tracking
```bash
https://dagshub.com/iampraveens/Customer-Churn-Prediction-MLOps/experiments/
```
Here I've implemented `MLFlow` to track my models on `DagsHub`. Check out using about link

## Dockerized Web App
You can also deploy the Customer Churn Prediction web application using Docker. Build the Docker image and run the container:
```bash
docker build -t your_docker_username/telcom-churn-app:latest .
```
- To build a docker image.

```bash
docker run -d -p 8501:8501 your_docker_username/telcom-churn-app
```
- To run as a container.

Access the web app at `http://localhost:8501` or `your_ip_address:8501` in your web browser.
Else if you want to access my pre-built container, here is the code below to pull from docker hub(Public).
```bash
docker pull iampraveens/telcom-churn-app:latest
```
## License 
This project is licensed under the MIT License - see the [License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt) file for details.

This README provides an overview of the project, its structure, how to get started, how to train the model, make predictions, tracking the model and even deploy a Dockerized web app.
