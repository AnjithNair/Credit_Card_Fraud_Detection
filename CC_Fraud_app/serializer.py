from rest_framework import serializers
from .models import FileName, Accuracies, ConfusionMatrix


class AccuracySerializer(serializers.ModelSerializer):
    class Meta:
        model = Accuracies
        fields = "__all__"


class ConfusionMatrixSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConfusionMatrix
        fields = "__all__"


class FileNameSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileName
        fields = "__all__"
