#!/user/bin/python
#"utf-8"

import numpy as np
import scipy as sp
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()

features = data.data
features_name = data.feature_names
target = data.target
target_names = data.target_names

k=0
for i in range(4):
	for j in range(i+1,4):
		k=k+1
		plt.subplot(2,3,k,axisbg='pink')
		for t,marker,color in zip(xrange(3),'>ox','rgb'):
			plt.scatter(features[target==t,i],features[target==t,j],marker=marker,c=color,label=target_names[t])			
			plt.legend(loc="upper left")
			plt.xlabel(features_name[i])
			plt.ylabel(features_name[j])		
			plt.autoscale()
			plt.grid()

					


plt.show()

