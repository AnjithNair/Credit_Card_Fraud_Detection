from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .credit_card_fraud_detection import CCD_Fraud_Detection
from rest_framework.response import Response


def index(request):
    return render(request, 'Index.html')

def graph(request):
    rf = request.GET.get('rf', None)
    lr = request.GET.get('lr', None)
    svm = request.GET.get('svm', None)

    context = {
        'rf': rf,
        'lr': lr,
        'svm': svm,
    }

    return render(request, 'graph.html', context)


@api_view(['POST'])
def getdata(request):
    data = request.data
    file_name = data['file_name']

    CC = CCD_Fraud_Detection(file_name)
    value1 = CC.LogisticRegression_for_cc_fraud()
    value3 = CC.SVM_for_cc_fraud()
    value2 = CC.random_forest_for_cc_fraud()
    Accuracy={
        "Accuracy of LogisticRegression": value1,
        "Accuracy of RandomForest": value3,
        "Accuracy of SVM Model" : value2
    }
    return Response(Accuracy)