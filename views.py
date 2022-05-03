from django.shortcuts import render

import numpy as np
import pandas as pd
from requests import request
from .apps import ApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response

class Prediction(APIView):
    def post(self, request):     
        number_emergency = request.GET.get('numemer',None)
        number_inpatient = request.GET.get('numinp',None)
        number_diagnoses = request.GET.get('numdiag',None)
        repaglinide = request.GET.get('repagl',None)
        tolbutamide = request.GET.get('tol',None)
        pioglitazone = request.GET.get('pio',None)
        acarbose = request.GET.get('acar',None)
        miglitol = request.GET.get('migl',None)
        glyburide_etformin = request.GET.get('gly',None)
        glipizide_metformin = request.GET.get('glip',None)
        diabetesMed = request.GET.get('diabmed',None)
        j=[[0,0,0,0,0,0,0,0,0,0,0,0]]
        p=int([number_emergency, number_inpatient, number_diagnoses, repaglinide, tolbutamide,pioglitazone,acarbose,miglitol,glyburide_etformin,glipizide_metformin,diabetesMed])
        j.append(list(p))
        j=np.array(j)
        kmen = ApiConfig.model
        PredictionMade = kmen.fit_predict(j)
        response_dict = {"Readmitted_NO": PredictionMade[1]}
        return Response(response_dict, status=200)

       





