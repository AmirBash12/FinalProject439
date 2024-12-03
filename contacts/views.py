from django.shortcuts import render, get_object_or_404, redirect
from .models import Contact
from django.db.models import Q
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from django.shortcuts import render
from contacts.models import Contact
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')
nltk.data.path.append('/path/to/your/nltk_data')
from sklearn.impute import SimpleImputer
from django.http import HttpResponseBadRequest
# Download necessary NLTK resources

def contact_list(request):
    sort_by = request.GET.get('sort_by', "")
    order = request.GET.get('order', "asc")  # Ascending or descending

    contacts = Contact.objects.all()

    if sort_by:
        direction = '' if order == 'asc' else '-'
        contacts = contacts.order_by(f'{direction}{sort_by}')

    return render(request, 'contacts/contact_list.html', {
        'contacts': contacts,
        'sort_by': sort_by,
        'order': order,
    })



def contact_detail(request, pk):
    contact = get_object_or_404(Contact, pk=pk)
    return render(request, 'contacts/contact_detail.html', {'contact': contact})

from django.http import HttpResponseBadRequest

def sanitize_phone_number(phone):
    """Remove any leading single quote or extra spaces."""
    return phone.strip().lstrip("'")

def contact_create(request):
    specializations = Contact.objects.values_list('specialization', flat=True).distinct()
    cities = Contact.objects.values_list('city', flat=True).distinct()

    if request.method == 'POST':
        name = request.POST['name']
        specialization = request.POST['specialization']
        city = request.POST['city']
        address = request.POST['address']
        fees = request.POST['fees']
        rating = request.POST.get('rating', 3.0)  # Default rating if not provided
        phone = sanitize_phone_number(request.POST['phone'])

        # Validation checks
        if float(fees) >= 2000:
            return HttpResponseBadRequest("Fees must be less than $2000.")
        if not phone.startswith(("01", "02", "03", "04", "05", "06", "07", "08", "09", "76", "81")) or len(phone) != 8:
            return HttpResponseBadRequest("Invalid phone number. It must be a valid Lebanese number.")

        # Create the contact
        Contact.objects.create(
            name=name,
            specialization=specialization,
            city=city,
            address=address,
            fees=fees,
            rating=rating,
            phone=phone,
        )
        return redirect('contact_list')

    return render(request, 'contacts/contact_form.html', {
        'specializations': specializations,
        'cities': cities,
    })

def contact_update(request, pk):
    contact = get_object_or_404(Contact, pk=pk)
    specializations = Contact.objects.values_list('specialization', flat=True).distinct()
    cities = Contact.objects.values_list('city', flat=True).distinct()

    if request.method == 'POST':
        # Update fields only if new values are provided
        contact.name = request.POST.get('name', contact.name)
        contact.specialization = request.POST.get('specialization', contact.specialization)
        contact.city = request.POST.get('city', contact.city)
        contact.address = request.POST.get('address', contact.address)
        contact.fees = request.POST.get('fees', contact.fees)
        contact.rating = request.POST.get('rating', contact.rating)
        contact.phone = sanitize_phone_number(request.POST.get('phone', contact.phone))

        # Validation checks
        if float(contact.fees) >= 2000:
            return HttpResponseBadRequest("Fees must be less than $2000.")
        if not contact.phone.startswith(("01", "02", "03", "04", "05", "06", "07", "08", "09", "76", "81")) or len(contact.phone) != 8:
            return HttpResponseBadRequest("Invalid phone number. It must be a valid Lebanese number.")

        # Save the updated contact
        contact.save()
        return redirect('contact_list')

    return render(request, 'contacts/contact_form.html', {
        'contact': contact,
        'specializations': specializations,
        'cities': cities,
    })


def contact_delete(request, pk):
    contact = get_object_or_404(Contact, pk=pk)
    if request.method == 'POST':
        contact.delete()
        return redirect('contact_list')
    return render(request, 'contacts/contact_confirm_delete.html', {'contact': contact})


def recommend_surgeons(request):
    # Fetch dropdown options dynamically
    specializations = Contact.objects.values_list('specialization', flat=True).distinct()
    cities = Contact.objects.values_list('city', flat=True).distinct()

    # Get filters and sorting options from the request
    specialization = request.GET.get('specialization', "Any")
    city = request.GET.get('city', "Any")
    max_fees = request.GET.get('max_fees', "")
    min_rating = request.GET.get('min_rating', "")
    sort_by = request.GET.get('sort_by', "")
    order = request.GET.get('order', "asc")  # Ascending or descending
    user_city = request.GET.get('user_city', "Any")  # For proximity sorting

    # Filter contacts
    surgeons = Contact.objects.all()
    if specialization != "Any":
        surgeons = surgeons.filter(specialization__iexact=specialization)
    if city != "Any":
        surgeons = surgeons.filter(city__iexact=city)
    if max_fees:
        surgeons = surgeons.filter(fees__lte=max_fees)
    if min_rating:
        surgeons = surgeons.filter(rating__gte=min_rating)

    # Apply sorting
    if sort_by:
        direction = '' if order == 'asc' else '-'
        surgeons = surgeons.order_by(f'{direction}{sort_by}')
    elif sort_by == 'proximity' and user_city != "Any":
        surgeons = sorted(
            surgeons,
            key=lambda c: (0 if c.city == user_city else 1, c.city)
        )

    return render(request, 'contacts/recommendations.html', {
        'surgeons': surgeons,
        'specializations': ["Any"] + list(specializations),
        'cities': ["Any"] + list(cities),
        'sort_by': sort_by,
        'specialization': specialization,
        'city': city,
        'user_city': user_city,
        'max_fees': max_fees,
        'min_rating': min_rating,
        'order': order,
    })

import re

def extract_keywords(user_input):
    # Predefined lists for matching
    valid_specializations = [
        "ENT Specialist", "Neurosurgeon", "Pediatric Surgeon", "Cardiologist",
        "Gastroenterologist", "Orthopedic Surgeon", "Dermatologist",
        "Oncologist", "General Surgeon", "Plastic Surgeon"
    ]
    valid_cities = [
        "Beirut", "Tripoli", "Sidon", "Byblos", "Zahle", "Baalbek",
        "Jezzine", "Keserwan", "Jounieh", "Aley", "Chouf", "Tyre", "Batroun", "Zgharta", "Baabda"
    ]

    # Initialize extracted keywords with default values
    matched_keywords = {
        "specialization": "Any",
        "city": "Any",
        "min_rating": 1,
        "max_fees": 2000
    }

    # Tokenize input
    tokens = user_input.lower().split()

    for i, token in enumerate(tokens):
        # Match specialization
        for specialization in valid_specializations:
            if specialization.lower() in user_input.lower():
                matched_keywords["specialization"] = specialization
                break

        # Match city
        for city in valid_cities:
            if city.lower() in user_input.lower():
                matched_keywords["city"] = city
                break

        # Match rating
        if "rating" in token:
            try:
                # Check for "equal" or number directly after "rating"
                if "equal" in tokens[i:i + 3]:
                    index = tokens.index("equal", i)
                    matched_keywords["min_rating"] = int(tokens[index + 1])
                elif "=" in token:
                    matched_keywords["min_rating"] = int(token.split('=')[1])
                elif i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    matched_keywords["min_rating"] = int(tokens[i + 1])
            except (ValueError, IndexError):
                continue

        # Match cost
        if "cost" in token:
            try:
                # Check for "equal" or number directly after "cost"
                if "equal" in tokens[i:i + 3]:
                    index = tokens.index("equal", i)
                    matched_keywords["max_fees"] = int(tokens[index + 1])
                elif "=" in token:
                    matched_keywords["max_fees"] = int(token.split('=')[1])
                elif i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    matched_keywords["max_fees"] = int(tokens[i + 1])
            except (ValueError, IndexError):
                continue

    return matched_keywords


 

def smart_recommendation(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input', "")

        # Extract keywords
        keywords = extract_keywords(user_input)

        # Get filters from extracted keywords
        specialization = keywords["specialization"]
        city = keywords["city"]
        min_rating = keywords["min_rating"]
        max_fees = keywords["max_fees"]

        # Filter contacts based on extracted keywords
        contacts = Contact.objects.all()
        if specialization != "Any":
            contacts = contacts.filter(specialization__iexact=specialization)
        if city != "Any":
            contacts = contacts.filter(city__iexact=city)
        contacts = contacts.filter(rating__gte=min_rating, fees__lte=max_fees)

        return render(request, 'contacts/smart_recommendation.html', {
            'contacts': contacts,
            'user_input': user_input,
            'keywords': keywords,
        })

    return render(request, 'contacts/smart_form.html')



from matplotlib import pyplot as plt
import io
import base64
from django.http import HttpResponse
def extract_query_vector(user_input, knn_data, encoders):
    # Extract keywords using your existing `extract_keywords` function
    keywords = extract_keywords(user_input)

    # Encode specialization and city
    specialization = keywords["specialization"]
    city = keywords["city"]

    specialization_encoded = (
        encoders['specialization'].transform([specialization])[0]
        if specialization != "Any" else -1
    )
    city_encoded = (
        encoders['city'].transform([city])[0]
        if city != "Any" else -1
    )

    # Construct query vector
    query_vector = np.array([
        [
            specialization_encoded,
            city_encoded,
            keywords["max_fees"],
            keywords["min_rating"]
        ]
    ])

    # Handle NaN in the query vector
    query_vector = np.nan_to_num(query_vector, nan=0.0)

    return query_vector, keywords

def calculate_similarity(query_vector, knn_data):
    X = knn_data[['specialization', 'city', 'fees', 'rating']].values
    distances = np.linalg.norm(X - query_vector, axis=1)
    similarities = 1 / (1 + distances)  # Convert distance to similarity
    return similarities




from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer  # Use TF-IDF for better differentiation
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def similarity_recommendation(request):
    if request.method == "POST":
        user_input = request.POST.get("user_input", "").lower()

        # Fetch all contacts
        contacts = Contact.objects.all()
        df = pd.DataFrame(contacts.values())

        if df.empty:
            return render(request, 'contacts/similarity_recommendation.html', {
                'contacts_with_scores': [],
                'user_input': user_input,
            })

        # Extract keywords
        keywords = extract_similarity_keywords(user_input)
        user_city = keywords.get("city", "Any").lower()
        user_specialization = keywords.get("specialization", "Any").lower()
        user_max_fees = keywords.get("max_fees", None)
        user_min_rating = keywords.get("min_rating", None)

        # Normalize numerical features (fees and rating)
        df["fees"] = df["fees"].astype(float)
        df["rating"] = df["rating"].astype(float)

        # City and Specialization Matching
        df["city_match"] = (df["city"].str.lower() == user_city).astype(int)
        df["specialization_match"] = (df["specialization"].str.lower() == user_specialization).astype(int)

        # Calculate Fees Closeness
        if user_max_fees is not None:
            df["fees_closeness"] = 1 - (abs(df["fees"] - user_max_fees) / user_max_fees)
            df["fees_closeness"] = df["fees_closeness"].clip(lower=0)  # Prevent negative values
        else:
            df["fees_closeness"] = 0  # If no cost specified, fees closeness contributes nothing

        # Calculate Rating Closeness
        if user_min_rating is not None:
            df["rating_closeness"] = 1 - (abs(df["rating"] - user_min_rating) / 5)
            df["rating_closeness"] = df["rating_closeness"].clip(lower=0)  # Prevent negative values
        else:
            df["rating_closeness"] = 0  # If no rating specified, rating closeness contributes nothing

        # Final Similarity Score Calculation
        df["similarity_score"] = (
            0.28 * df["city_match"] +  # Match city (high weight for exact match)
            0.28 * df["specialization_match"] +  # Match specialization (high weight for exact match)
            0.22 * df["fees_closeness"] +  # Match fees (moderate weight for closeness)
            0.22 * df["rating_closeness"]  # Match rating (moderate weight for closeness)
        )

        # Sort by similarity score
        df = df.sort_values(by="similarity_score", ascending=False)

        # Fetch the top 10 contacts
        top_10_contacts = df.iloc[:10]

        # Combine contacts with similarity scores
        contacts_with_scores = list(
            zip(
                Contact.objects.filter(pk__in=top_10_contacts["id"].tolist()),
                top_10_contacts["similarity_score"].tolist(),
            )
        )

        return render(request, 'contacts/similarity_recommendation.html', {
            'contacts_with_scores': contacts_with_scores,
            'user_input': user_input,
        })

    return render(request, "contacts/similarity_form.html")







def extract_similarity_keywords(user_input):
    # Predefined lists for matching
    valid_specializations = [
        "ENT Specialist", "Neurosurgeon", "Pediatric Surgeon", "Cardiologist",
        "Gastroenterologist", "Orthopedic Surgeon", "Dermatologist",
        "Oncologist", "General Surgeon", "Plastic Surgeon"
    ]
    valid_cities = [
        "Beirut", "Tripoli", "Sidon", "Byblos", "Zahle", "Baalbek",
        "Jezzine", "Keserwan", "Jounieh", "Aley", "Chouf", "Tyre", "Batroun", "Zgharta", "Baabda"
    ]

    # Initialize extracted keywords with default values
    matched_keywords = {
        "specialization": "Any",
        "city": "Any",
        "min_rating": 1,  # Default minimum rating
        "max_fees": 2000  # Default maximum fees
    }

    # Tokenize input
    tokens = user_input.lower().split()

    for i, token in enumerate(tokens):
        # Match specialization
        for specialization in valid_specializations:
            if specialization.lower() in user_input.lower():
                matched_keywords["specialization"] = specialization
                break

        # Match city
        for city in valid_cities:
            if city.lower() in user_input.lower():
                matched_keywords["city"] = city
                break

        # Match rating (handle variations like "rating=3", "rating equal 3", "rating 3")
        if "rating" in token:
            try:
                # Check for "equal" or number directly after "rating"
                if "equal" in tokens[i:i + 3]:
                    index = tokens.index("equal", i)
                    matched_keywords["min_rating"] = int(tokens[index + 1])
                elif "=" in token:
                    matched_keywords["min_rating"] = int(token.split('=')[1])
                elif i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    matched_keywords["min_rating"] = int(tokens[i + 1])
            except (ValueError, IndexError):
                continue

        # Match cost (handle variations like "cost=1500", "cost equal 1500", "cost 1500")
        if "cost" in token:
            try:
                # Check for "equal" or number directly after "cost"
                if "equal" in tokens[i:i + 3]:
                    index = tokens.index("equal", i)
                    matched_keywords["max_fees"] = int(tokens[index + 1])
                elif "=" in token:
                    matched_keywords["max_fees"] = int(token.split('=')[1])
                elif i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    matched_keywords["max_fees"] = int(tokens[i + 1])
            except (ValueError, IndexError):
                continue

    return matched_keywords










from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Function to train the KNN model and return necessary data
def train_knn_model():
    contacts = Contact.objects.all().values()
    df = pd.DataFrame(contacts)

    # Encode categorical variables
    label_encoders = {
        'specialization': LabelEncoder(),
        'city': LabelEncoder(),
    }
    for col, le in label_encoders.items():
        if col in df.columns and not df[col].isnull().all():
            df[col] = le.fit_transform(df[col].fillna("Unknown"))
        else:
            df[col] = -1  # Handle missing categorical data with a default value

    # Features for training
    features = ['specialization', 'city', 'fees', 'rating']

    # Handle missing values in numerical columns
    imputer = SimpleImputer(strategy='mean')
    df[features] = imputer.fit_transform(df[features])

    X = df[features]

    # Train KNN
    knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
    knn.fit(X)

    return knn, df, label_encoders

# Train the model and initialize global variables
knn_model, knn_data, encoders = train_knn_model()

