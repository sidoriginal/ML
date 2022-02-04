from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
dat=iris.data
tg=iris.target

clf=KNeighborsClassifier()

clf.fit(dat,tg)

ans=['Iris setosa','Iris virginica','Iris versicolor']

p=clf.predict([[32,0,1,2]])

print(ans[p[0]])