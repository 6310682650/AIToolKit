from django.db import models

# class TrainingData(models.Model):
#     data = models.TextField()

# class PredictionResult(models.Model):
#     result = models.CharField(max_length=100)

class CardiacDiseaseRisk(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    weight = models.FloatField()
    smoker = models.BooleanField()
    diabetic = models.BooleanField()

    def __str__(self):
        return f"Cardiac Disease Risk - ID: {self.id}"
