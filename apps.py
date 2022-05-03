import os
import joblib
from django.apps import AppConfig
from django.conf import settings


class ApiConfig(AppConfig):
    name = 'monitor'
    MODEL_FILE = '/Users/rishabhchauhan/Desktop/djangoapi/mainapp/ml/model/predictmod.joblib'
    model = joblib.load(MODEL_FILE)