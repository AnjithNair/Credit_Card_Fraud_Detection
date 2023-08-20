from django.contrib import admin

from .models import *
from django.apps import apps

# # Register your models here.
admin.site.register(FileName)
admin.site.register(Accuracies)
admin.site.register(ConfusionMatrix)

# models = apps.get_models()

# for model in models:
#     admin.site.register(model)
