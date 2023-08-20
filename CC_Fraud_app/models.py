from django.db import models

# Create your models here.


class FileName(models.Model):
    class Files(models.TextChoices):
        CCKaggle = "CC", "CCKAGGLE"
        CCProcess = "CCP", "CCPROCESSED"

    file_name = models.CharField(choices=Files.choices, max_length=10)


class Accuracies(models.Model):
    FileName = models.ForeignKey(FileName, on_delete=models.CASCADE)
    RandomForest = models.FloatField(null=True, blank=True)
    SVM = models.FloatField(null=True, blank=True)
    LogisticRegression = models.FloatField(null=True, blank=True)


class ConfusionMatrix(models.Model):
    FileName = models.ForeignKey(FileName, on_delete=models.CASCADE)
    CM_LR_TP = models.IntegerField(null=True, blank=True)
    CM_LR_FP = models.IntegerField(null=True, blank=True)
    CM_LR_FN = models.IntegerField(null=True, blank=True)
    CM_LR_TN = models.IntegerField(null=True, blank=True)

    CM_RF_TP = models.IntegerField(null=True, blank=True)
    CM_RF_FP = models.IntegerField(null=True, blank=True)
    CM_RF_FN = models.IntegerField(null=True, blank=True)
    CM_RF_TN = models.IntegerField(null=True, blank=True)

    CM_SVM_TP = models.IntegerField(null=True, blank=True)
    CM_SVM_FP = models.IntegerField(null=True, blank=True)
    CM_SVM_FN = models.IntegerField(null=True, blank=True)
    CM_SVM_TN = models.IntegerField(null=True, blank=True)
