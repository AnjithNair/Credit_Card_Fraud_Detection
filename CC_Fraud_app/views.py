from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .credit_card_fraud_detection import CCD_Fraud_Detection
from rest_framework.response import Response


def index(request):
    return render(request, 'Index.html')

from django.shortcuts import redirect, reverse

def graph(request):
    if request.method == 'POST':
        rf = request.POST.get('rf')
        lr = request.POST.get('lr')
        svm = request.POST.get('svm')
        lr_cm = request.POST.get('lr_cm')
        rf_cm = request.POST.get('rf_cm')
        svm_cm = request.POST.get('svm_cm')
        
        context = {
            'rf': rf,
            'lr': lr,
            'svm': svm,
        }

        return redirect(reverse('confusionmatrix') + f'?rf={rf}&lr={lr}&svm={svm}&lr_cm[0]={lr_cm[0]}&lr_cm[1]={lr_cm[1]}&lr_cm[2]={lr_cm[2]}&lr_cm[3]={lr_cm[3]}&rf_cm[0]={rf_cm[0]}&rf_cm[1]={rf_cm[1]}&rf_cm[2]={rf_cm[2]}&rf_cm[3]={rf_cm[3]}&svm_cm[0]={svm_cm[0]}&svm_cm[1]={svm_cm[1]}&svm_cm[2]={svm_cm[2]}&svm_cm[3]={svm_cm[3]}')



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
    lr_cm = [int(request.GET.get('lr_cm[0]', 0)), int(request.GET.get('lr_cm[1]', 0)), int(request.GET.get('lr_cm[2]', 0)), int(request.GET.get('lr_cm[3]', 0))]
    rf_cm = [int(request.GET.get('rf_cm[0]', 0)), int(request.GET.get('rf_cm[1]', 0)), int(request.GET.get('rf_cm[2]', 0)), int(request.GET.get('rf_cm[3]', 0))]
    svm_cm = [int(request.GET.get('svm_cm[0]', 0)), int(request.GET.get('svm_cm[1]', 0)), int(request.GET.get('svm_cm[2]', 0)), int(request.GET.get('svm_cm[3]', 0))]

    context = {
        'lr_cm': lr_cm,
        'rf_cm': rf_cm,
        'svm_cm': svm_cm,
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