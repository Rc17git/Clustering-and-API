from django.urls import path
import monitor.views as views

urlpatterns = [
    path('result', views.Prediction.as_view()),
]