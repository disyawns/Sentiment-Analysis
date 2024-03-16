# Sentiment-Analysis

# Flask Sentiment Analysis Web App

This is a Flask web application for sentiment analysis, where users can input text and get predictions about the sentiment of the input text. The application is hosted on AWS Elastic Beanstalk.

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The project consists of a Flask web application that uses a machine learning model to predict the sentiment of text input by users. The application preprocesses the text input, transforms it into TF-IDF features, and uses a trained machine learning model to make predictions. The predicted sentiment (negative, neutral, or positive) is then displayed to the user.

## Getting Started

### Prerequisites

Before running the application locally or deploying it to AWS Elastic Beanstalk, make sure you have the following prerequisites installed:

- Python 3.x
- Flask
- pandas
- scikit-learn
- nltk

You can install these dependencies using pip:

```bash
pip install flask pandas scikit-learn nltk
Installation
To run the application locally, follow these steps:

Clone this repository to your local machine:

git clone https://github.com/disyawns/Sentiment-Analysis.git
Navigate to the project directory:

cd your-repository
Run the Flask application:

python app.py
The application will be accessible at http://localhost:5000 in your web browser.

Usage
Once the application is running, open it in your web browser and input text into the provided text box. Submit the text to get predictions about its sentiment. The application will display the predicted sentiment (negative, neutral, or positive).

Deployment
The application can be deployed to AWS Elastic Beanstalk for production use. Follow the instructions provided in the deployment section of this README to deploy the application to AWS Elastic Beanstalk.

Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository
Create a new branch (git checkout -b feature/new-feature)
Make your changes
Commit your changes (git commit -am 'Add new feature')
Push to the branch (git push origin feature/new-feature)
Create a new pull request
License
This project is licensed under the MIT License.
