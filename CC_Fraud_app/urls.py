from django.urls import path
from .views import index, getdata

urlpatterns = [
    path('', index, name='index'),
    path('getdata', getdata, name='getdata'),
]
