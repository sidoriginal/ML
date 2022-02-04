import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diab=datasets.load_diabetes()
#print(diab.keys())
#dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
#print(diab.data)
#print(diab.DESCR)

#diab_x=diab.data[:,np.newaxis,2] //for 1 data point
diab_x=diab.data #//for all data points
diab_x_tr=diab_x[-20:]
diab_x_test=diab_x[:-20]

# diab_x_tr2=np.array[[1],[2],[3]] //using numpy array


diab_y_tr=diab.target[-20:]
diab_y_test=diab.target[:-20]

model=linear_model.LinearRegression();
model.fit(diab_x_tr,diab_y_tr)

diab_y_pr=model.predict(diab_x_test)
 
print("Error is:",mean_squared_error(diab_y_test,diab_y_pr))

print("Coeffecients:",model.coef_)
print("Intercepts",model.intercept_)

# plt.plot(diab_x_test,diab_y_pr)    
# plt.scatter(diab_x_test,diab_y_test)    //applicable to 1 data point
# plt.show()