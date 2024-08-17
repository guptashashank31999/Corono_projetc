from django.db import models

# Create your models here.


class RegisterModel(models.Model):
    firstname=models.CharField(max_length=300)
    lastname=models.CharField(max_length=200)
    userid=models.CharField(max_length=200)
    password=models.CharField(max_length=200)
    phoneno=models.BigIntegerField()
    email=models.EmailField(max_length=400)
    gender=models.CharField(max_length=200)


class DetectionModel(models.Model):
    covid_Userd = models.ForeignKey(RegisterModel)
    analysisvalue = models.CharField(max_length=5000)
    image = models.FileField()