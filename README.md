# Contact List Management System

This project is a Contact List Management System, developed as part of the EECE 439 course. It features CRUD operations, smart search, and a similarity-based recommendation system powered by Natural Language Processing (NLP) and machine learning algorithms.

## Features

### 1. CRUD Operations
- Add, update, delete, and view contacts.
- Each contact includes fields such as:
  - Name
  - Specialization
  - City
  - Fees
  - Rating
  - Phone Number

### 2. Smart Search
- Allows users to enter queries in natural language to filter and sort contacts.
- Extracts keywords like specialization, city, cost, and rating using NLP.
- Matches extracted inputs to database fields and returns relevant results.

### 3. Similarity-Based Recommendations
- Recommends contacts based on user input.
- Uses NLP to extract relevant keywords and numerical data.
- Employs the K-Nearest Neighbors (KNN) algorithm to compute similarity scores.
- Dynamically ranks recommendations based on:
  - City
  - Specialization
  - Numerical proximity to cost and rating preferences.

## Technologies Used

### Backend
- Python
- Django Framework

### Frontend
- HTML, CSS, JavaScript
- Chart.js for visualization

### Machine Learning
- Scikit-learn for KNN algorithm and feature normalization.

### NLP
- NLTK for keyword extraction and stopword removal.

### Deployment
- Deployed on Azure using the Django framework.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/contact-list-management.git
   cd contact-list-management
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Apply migrations:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. Run the development server:
   ```bash
   python manage.py runserver
   ```

6. Access the application at `http://127.0.0.1:8000/`.

## Deployment on Azure

### Prerequisites
- Install the Azure CLI and Azure extension for VS Code.
- Create an Azure Web App.

### Steps
1. Generate a `requirements.txt` file:
   ```bash
   pip freeze > requirements.txt
   ```

2. Collect static files:
   ```bash
   python manage.py collectstatic
   ```

3. Initialize a Git repository and push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/contact-list-management.git
   git push -u origin main
   ```

4. Link your Azure Web App to your GitHub repository.

5. Configure environment variables in Azure:
   - `DEBUG`: `False`
   - `ALLOWED_HOSTS`: `<your-app-name>.azurewebsites.net`

6. Deploy the app through Azure.

## Example Usage

### Smart Search
- Input: `"Cardiologist in Beirut with rating equal 4 and cost equal 1500"`
- Output: A list of contacts matching the specified criteria.

### Similarity-Based Recommendations
- Input: `"Pediatric Surgeon in Tripoli with cost 1000 and rating 3"`
- Output: Contacts ranked by similarity score, visualized in a bar chart.

