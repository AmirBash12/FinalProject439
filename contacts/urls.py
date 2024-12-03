from django.urls import path
from . import views

urlpatterns = [
    path('', views.contact_list, name='contact_list'),  # Main contact list
    path('<int:pk>/', views.contact_detail, name='contact_detail'),  # View contact details
    path('new/', views.contact_create, name='contact_create'),  # Add a new contact
    path('<int:pk>/edit/', views.contact_update, name='contact_update'),  # Edit a contact
    path('<int:pk>/delete/', views.contact_delete, name='contact_delete'),  # Delete a contact
    path('recommendations/', views.recommend_surgeons, name='recommend_surgeons'),  # Filter-based recommendations
    path('smart-recommendation/', views.smart_recommendation, name='smart_recommendation'),
    path('similarity-recommendation/', views.similarity_recommendation, name='similarity_recommendation'),
]

