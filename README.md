# X-rays COVID-nonCOVID classifier app

This is a machine learning application that classifies X-ray images as COVID-positive or non-COVID based on trained models. This is a demo prototype for detection of COVID-19 cases using X-ray imaging.

## Disclaimer
Please note that this application is for demonstration purposes only and should not be used as a substitute for professional medical diagnosis or advice. The classification results provided by the app are based on trained machine learning models, which may have limitations and false positives/negatives. Always consult with medical experts and trusted healthcare professionals for accurate diagnosis, medical guidance, and treatment.

The developers and contributors of this application are not responsible for any misuse or misinterpretation of the classification results or any direct or indirect consequences arising from the use of this application. The app should be used responsibly and in conjunction with proper medical expertise.

## Features

- X-ray Classification: The app uses machine learning models to classify X-ray images as either COVID-positive or non-COVID.
- Web Interface: It provides a user-friendly web interface where users can upload X-ray images and receive classification results.
- Deployment-ready: The app is designed for easy deployment, allowing users to set it up quickly and start using it with minimal configuration.

## Technologies

- 🟪 TypeScript
- ⚛️ React
- 🔷 Node.js
- 📦 Docker
- 🌊 Git

## Usage
Upload an X-ray image using the provided interface.
Click the "Classify" button to initiate the classification process.
Wait for the results to appear, indicating whether the X-ray image is COVID-positive or non-COVID.

## License
This project is licensed under the MIT License.
Feel free to customize the above description to match the specifics of your X-rays COVID-nonCOVID classifier app, and include any additional sections or details that are relevant to your project.

## Prerequisites

- Docker

## Getting Started

To get the project running locally, follow these steps:

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/portfolio-app.git
   cd portfolio-app


## Setup
* Starting  from base_config branch
Run next commands:
```
docker-compose up -d --build
```

## Configuration

    |--app
    |--app/backend
    |--app/frontend

## Backend

## Setup 
After image is build and running
```
docker exec -it backend-app django-admin startproject backend
```
Once configured our project let's create our first django-app (users)
```
docker exec -it django-app bash 
cd app && python manage.py startapp users
```
Then apply migrations and create superuser
```
docker exec -it django-app bash 
cd app && python manage.py makemigrations 
python manage.py migrate
python manage.py cretesuperuser
```
## Setup 
* Starting from master or main/dev branch
Run next:
```
docker-compose build
docker-compose up -d
```
Create new django-app using:
```
docker exec -it django-app python manage.py migrate
```

## Tailwind

## References:

* [React vs. Vue (Exact Todo App) By Example](https://medium.com/js-dojo/react-vs-vue-exact-todo-app-comparison-by-example-14cc56efc5e5)
