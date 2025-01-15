from django.db import models

# Create your models here.
from django.conf import settings
import os

class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='temp/')
    dataset = models.CharField(max_length=50)
    processed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.dataset} - {self.image.name}"