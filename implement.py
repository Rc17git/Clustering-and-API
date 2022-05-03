import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
data=pd.read_csv("TRAIN.csv")

data=data.drop(columns=['weight','payer_code','medical_specialty','A1Cresult','max_glu_serum',"admission_type_id",'admission_source_id','discharge_disposition_id'],axis=1)

for i in range(len(data["gender"])):
    if data.iloc[i,1]=='Unknown/Invalid':
        x=i
data=data.drop(x)
data=data.reset_index(drop=True)

col=['race','gender','age','metformin', 'repaglinide',
       'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
       'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
       'glipizide-metformin', 'glimepiride-pioglitazone',
       'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
       'diabetesMed']
for i in col:
    data[i]=data[i].astype('category')
    data[i]=data[i].cat.codes

for m in range(1,4):
    for i in range(len(data["diag_3"])):
        if data.loc[i,"diag_"+str(m)]=="?":
            data.loc[i,"diag_"+str(m)]=None
    data["diag_"+str(m)]=pd.to_numeric(data["diag_"+str(m)],errors='coerce')
    data["diag_"+str(m)]=data["diag_"+str(m)].fillna(data["diag_"+str(m)].mean())

colly=['race', 'gender', 'age','time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
       'diag_2', 'diag_3', 'number_diagnoses', 'metformin', 'repaglinide',
       'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
       'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
       'glipizide-metformin', 'glimepiride-pioglitazone',
       'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
       'diabetesMed']
for i in colly:
    for j in range(71235):
        if data.loc[j,i]=="?":
            data.loc[j,i]=None

X=data.drop(['readmitted_NO'],axis=1)
Y=data['readmitted_NO']

model = LogisticRegression()
rfe = RFE(model,n_features_to_select=15)
fit = rfe.fit(X, Y)
selec=fit.support_
c=X.columns
dropf=[]
for i in range(0,38):
    if selec[i]==False:
        dropf.append(c[i])
newX=X.drop(dropf,axis=1)

newkm=KMeans(2)
newkm.fit(newX)
newpredy=newkm.fit_predict(newX)
correct_labels1 = sum(Y == newpredy)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels1, Y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels1/float(Y.size)))