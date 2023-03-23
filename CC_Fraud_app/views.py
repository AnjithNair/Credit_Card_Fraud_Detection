from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Create your views here.
@api_view(['GET'])
def getdata(request,date):
	print(date)
	# date = data['date_time']
	weather={
		"aqi":"85"
	}
	return Response(weather)