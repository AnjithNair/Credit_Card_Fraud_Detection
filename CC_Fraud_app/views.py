from django.shortcuts import render
from rest_framework.decorators import api_view
from .credit_card_fraud_detection import CCD_Fraud_Detection
from rest_framework.response import Response
# pip  install -r requirmets.txt


# Create your views here.
@api_view(['POST'])
def getdata(request):
	data = request.data
	file_name = data['file_name']

	CC = CCD_Fraud_Detection()
	value1 = CC.LogisticRegression_for_cc_fraud(file_name)
	value3 = CC.SVM_for_cc_fraud(file_name)
	value2 = CC.random_forest_for_cc_fraud(file_name)
	Accuracy={
		"Accuracy of LogisticRegression": value1,
		"Accuracy of RandomForest": value2,
		"Accuracy of SVM Model" : value3
				}
	return Response(Accuracy)