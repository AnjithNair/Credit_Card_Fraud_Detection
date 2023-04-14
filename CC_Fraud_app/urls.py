from django.urls import path
from .views import index, getdata,graph
from . import views
urlpatterns = [
    path('', index, name='index'),
    path('getdata/', getdata, name='getdata'),
    path('graph/', graph, name='graph'),
    path('graph.html', views.graph, name='graph')
]
