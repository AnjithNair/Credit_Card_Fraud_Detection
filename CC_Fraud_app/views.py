from django.shortcuts import render
from .credit_card_fraud_detection import CCD_Fraud_Detection
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import FileName, Accuracies, ConfusionMatrix
from .serializer import *

# from django.views.decorators.csrf import csrf_exempt
# from rest_framework.decorators import api_view


class CreditCardFraud(APIView):
    def graph(self, request):
        FileName_id = request.GET.get("filename_id")
        Accuracy = Accuracies.objects.filter(FileName=int(FileName_id)).values()

        return Response(Accuracy[0])

    def confusion_matrix(self, request):
        FileName_id = request.GET.get("filename_id")
        CM = ConfusionMatrix.objects.filter(FileName=int(FileName_id)).values()
        return Response(CM[0])

    def get(self, request, *args, **kwargs):
        if kwargs["page"] == "graph":
            return self.graph(request)

        if kwargs["page"] == "confusion-matrix":
            return self.confusion_matrix(request)

    def post(self, request, *args, **kwargs):
        if kwargs["page"] == "Index":
            file_name = request.data["file_name"]
            CC = CCD_Fraud_Detection(file_name)
            value1, lr_cm = CC.LogisticRegression_for_cc_fraud()
            value3, svm_cm = CC.SVM_for_cc_fraud()
            value2, rf_cm = CC.random_forest_for_cc_fraud()
            print(rf_cm)
            if file_name == "creditcard":
                serializer = FileNameSerializer(data={"file_name": "CC"})
            else:
                serializer = FileNameSerializer(data={"file_name": "CCP"})

            if serializer.is_valid():
                obj = serializer.save()
            else:
                return Response(
                    {"error": "Invalid FileName", "Serializer Error": serializer.errors}
                )

            Accuracy = {
                "LogisticRegression": float(value1),
                "RandomForest": float(value3),
                "SVM": float(value2),
                "FileName": obj.id,
            }

            Accuracyser = AccuracySerializer(data=Accuracy)

            if Accuracyser.is_valid():
                Accuracyser.save()
            else:
                return Response(
                    {
                        "error": "Invalid Accuracies",
                        "Serializer Error": Accuracyser.errors,
                    }
                )

            CM = {
                "CM_LR_TP": int(lr_cm[0][0]),
                "CM_LR_FP": int(lr_cm[0][1]),
                "CM_LR_FN": int(lr_cm[1][0]),
                "CM_LR_TN": int(lr_cm[1][1]),
                "CM_RF_TP": int(rf_cm[0][0]),
                "CM_RF_FP": int(rf_cm[0][1]),
                "CM_RF_FN": int(rf_cm[1][0]),
                "CM_RF_TN": int(rf_cm[1][1]),
                "CM_SVM_TP": int(svm_cm[0][0]),
                "CM_SVM_FP": int(svm_cm[0][1]),
                "CM_SVM_FN": int(svm_cm[1][0]),
                "CM_SVM_TN": int(svm_cm[1][1]),
                "FileName": obj.id,
            }

            cmserializer = ConfusionMatrixSerializer(data=CM)

            if cmserializer.is_valid():
                cmserializer.save()
            else:
                return Response(
                    {
                        "error": "Invalid Confusion Matrix",
                        "Serializer Error": cmserializer.errors,
                    }
                )

        return Response({"success": "Fraud Detection-Completed", "FileName_id": obj.id})
