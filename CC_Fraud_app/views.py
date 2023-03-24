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
	{
"file_name":"creditcard"
}
	CC = CCD_Fraud_Detection()
	value = CC.LogisticRegression_for_cc_fraud(file_name)
	
	Accuracy={
		"score":value
				}
	return Response(Accuracy)