from django.db import models

class Contact(models.Model):
    name = models.CharField(max_length=100)
    specialization = models.CharField(max_length=100)  # e.g., Surgeon, Dentist
    city = models.CharField(max_length=100)
    address = models.TextField()
    fees = models.DecimalField(max_digits=10, decimal_places=2)
    rating = models.FloatField()  # e.g., 4.5 out of 5
    phone = models.CharField(max_length=15)

    def __str__(self):
        return self.name
