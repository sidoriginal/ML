from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris=datasets.load_iris()
x=iris["data"][:,3:]
y=(iris["target"]==1).astype(np.int)

clf=LogisticRegression()

clf.fit(x,y)
ans=clf.predict([[3.6]])
print(ans)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
print(y_prob)
plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
plt.show()

