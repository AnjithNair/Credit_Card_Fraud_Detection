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

def confusionmatrix(request):
    rf_cm = request.GET.getlist('rf_cm[]', [])
    lr_cm = request.GET.getlist('lr_cm[]', [])
    svm_cm = request.GET.getlist('svm_cm[]', [])

    context = {
        'rf_cm': [int(x) for x in rf_cm],
        'lr_cm': [int(x) for x in lr_cm],
        'svm_cm': [int(x) for x in svm_cm],
    }

    return render(request, 'confusionmatrix.html', context)

@api_view(['POST'])
def getdata(request):
    data = request.data
    file_name = data['file_name']

    CC = CCD_Fraud_Detection(file_name)
    value1,lr_cm = CC.LogisticRegression_for_cc_fraud()
    value3,svm_cm = CC.SVM_for_cc_fraud()
    value2,rf_cm = CC.random_forest_for_cc_fraud()
    Accuracy={
        "Accuracy of LogisticRegression": value1,
        "Accuracy of RandomForest": value3,
        "Accuracy of SVM Model" : value2,
        "CM of LogisticRegression-TP" : lr_cm[0][0],
        "CM of LogisticRegression-FP" : lr_cm[0][1],
        "CM of LogisticRegression-FN" : lr_cm[1][0],
        "CM of LogisticRegression-TN" : lr_cm[1][1],
        "CM of RandomForest - TP " : rf_cm[0][0],
        "CM of RandomForest - FP " : rf_cm[0][1],
        "CM of RandomForest - FN " : rf_cm[1][0],
        "CM of RandomForest - TN " : rf_cm[1][1],
        "CM of SVM Model - TP" : svm_cm[0][0],
        "CM of SVM Model - FP" : svm_cm[0][1],
        "CM of SVM Model - FN" : svm_cm[1][0],
        "CM of SVM Model - TN" : svm_cm[1][1]
    }
    return Response(Accuracy)