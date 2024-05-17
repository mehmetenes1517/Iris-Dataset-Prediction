import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

class Iris(nn.Module):
  def __init__(self):
    super().__init__()
    self.mdl1=nn.Sequential(
        nn.Linear(4,64),
        nn.ReLU(),
        nn.Linear(64,64),
        nn.ReLU(),
        nn.Linear(64,3)
    )
  def forward(self,x):
    return self.mdl1(x)

df=pd.read_csv('Iris.csv')
x=torch.FloatTensor(np.stack([df['SepalLengthCm'].values,df['SepalWidthCm'].values,df['PetalLengthCm'].values,df['PetalWidthCm'].values],axis=1).reshape(-1,4))
y=df['Species']
y.replace({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2},inplace=True)
y=torch.LongTensor(y.values.reshape(-1,1))


mdl1=Iris()
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(mdl1.parameters(),lr=0.001)
epochs=3000
losses=[]


for i in range(epochs):
  predict_y=mdl1(x)
  loss=criterion(predict_y,y.squeeze())
  losses.append(loss.item())

  #backprop
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

predictions=mdl1(x)
correct=0
for i in range(150):
  print(f'real : {y[i][0]} | predict : {predictions[i].argmax().item()}')
  if y[i][0]==predictions[i].argmax().item():
    correct+=1
print(f'accuracy : {100*(correct/150)}')



while True:
  a=float(input('SepalLengthCm :'))
  b=float(input('SepalWidthCm  :'))
  c=float(input('PetalLengthCm :'))
  d=float(input('PetalWidthCm  :'))
  if a==0 and  b==0 and c==0 and d==0:
    break
  pred=mdl1(torch.tensor([[a,b,c,d]]))[0].argmax().item()
  if pred==0:
    pred='Iris-setosa'
  elif pred==1:
    pred='Iris-versicolor'
  else:
    pred='Iris-virginica'

  print(f'{pred}')