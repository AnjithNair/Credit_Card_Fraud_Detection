from django.urls import path
from .views import index, getdata,graph,confusionmatrix
from . import views
urlpatterns = [
    path('', index, name='index'),
    path('getdata/', getdata, name='getdata'),
    path('graph/', graph, name='graph'),
    path('graph.html', views.graph, name='graph'),
    path('confusionmatrix/',confusionmatrix,name='confusionmatrix'),
    path('confusionmatrix.html', views.confusionmatrix, name='confusionmatrix')
]
