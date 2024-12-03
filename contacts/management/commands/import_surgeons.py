import csv
from django.core.management.base import BaseCommand
from contacts.models import Contact

class Command(BaseCommand):
    help = "Import surgeons from a CSV file"

    def handle(self, *args, **kwargs):
        file_path = r"C:\Users\ziadm\Downloads\formatted_lebanese_surgeon_sample_data.csv"  # Hardcoded path
        try:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    Contact.objects.create(
                        name=row['Name'],
                        specialization=row['Specialization'],
                        city=row['City'],
                        address=row['Address'],
                        fees=row['Fees'],
                        rating=row['Rating'],
                        phone=row['Phone'],
                    )
            self.stdout.write(self.style.SUCCESS("Successfully imported data"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error importing data: {e}"))
