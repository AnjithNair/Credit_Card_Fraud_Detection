from django.urls import path
from . import views

urlpatterns = [
    # path('', index, name='index'),
    # path('getdata/', getdata, name='getdata'),
    # path('graph/', graph, name='graph'),
    # path('graph.html', views.graph, name='graph'),
    # path('confusionmatrix/',views.confusionmatrix,name='confusionmatrix'),
    # path('confusionmatrix.html', views.confusionmatrix, name='confusionmatrix'),
    path(
        "CCD-fraud-detection/<str:page>/",
        views.CreditCardFraud.as_view(),
        name="Fraud-Detection",
    )
]
