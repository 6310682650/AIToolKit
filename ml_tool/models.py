from django.db import models

class TrainedModelHistory(models.Model):
    dataset = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    accuracy = models.FloatField()
    train_scale = models.FloatField()
    timestamp = models.DateTimeField()
    def __str__(self):
        return f"{self.dataset} - {self.model} - Accuracy: {self.accuracy}"
