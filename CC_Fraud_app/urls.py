from django.urls import path
from .views import index, getdata,graph

urlpatterns = [
    path('', index, name='index'),
    path('getdata', getdata, name='getdata'),
    path('graph/', graph, name='graph'),
]
