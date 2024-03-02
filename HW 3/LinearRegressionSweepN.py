import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error as rmse
from data_generator import postfix
from lift import lift
from liftDataset import lift_Dataset

# Number of samples
N = 8000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40


psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

# FOR Q3.
X = lift_Dataset(X)

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

NTrain = X_train.shape[0]
model = LinearRegression()

# Assuming the training data are already shuffled.
rmse_train = []
rmse_entire = []
rmse_test = []
frList = np.arange(0.02, 1.02, 0.02)
#frList = np.arange(0.1, 1.1, 0.1)
modelCoef = np.full((len(frList), X_test.shape[1]),np.nan)
for ifr,fr in enumerate(frList):    
    X_train1 = X_train[:np.round(NTrain*fr).astype(int),:]
    y_train1 = y_train[:np.round(NTrain*fr).astype(int)]
    #print("Fitting linear model...",end="")
    model.fit(X_train1, y_train1)
    modelCoef[ifr,:] = model.coef_
    #print(" done")
    rmse_train.append(rmse(y_train1,model.predict(X_train1)))
    #rmse_test = rmse(y_test,model.predict(X_test))
    rmse_entire.append(rmse(y,model.predict(X)))
    rmse_test.append(rmse(y_test,model.predict(X_test)))

fig = plt.figure(figsize=(6,2))
axRt = fig.add_subplot(1,3,1)
axRt.plot(np.round(frList * NTrain).astype(int),rmse_train)
axRt.set_xlabel('No. of training samples')
axRt.set_ylabel('RMSE on the training set')

# axRe = fig.add_subplot(1,3,2)
# axRe.plot(np.round(frList * NTrain).astype(int),rmse_entire)
# axRe.set_xlabel('No. of training samples')
# axRe.set_ylabel('RMSE on the entire set')

axRte = fig.add_subplot(1,3,2)
axRte.plot(np.round(frList * NTrain).astype(int),rmse_test)
axRte.set_xlabel('No. of training samples')
axRte.set_ylabel('RMSE on the test set')
#axRte.set_ylim(0,0.05)



axPar = fig.add_subplot(1,3,3)
for i in range(X_test.shape[1]):
    axPar.plot(np.round(frList * NTrain).astype(int),modelCoef[:,i], label="β%d" %i)
axPar.set_xlabel('No. of training samples')
axPar.set_ylabel('Model coefficients')
#axPar.legend(loc='best')

plt.tight_layout()
plt.show()
fig.savefig("Q3_N_8000_d_40_Sigma_0.01_Lift_detailed.png", bbox_inches="tight") # Adjust the name per assignment...
#plt.close

# Compute RMSE on train and test sets
    

#print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))


print("Model parameters:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    print(", β%d: %3.5f" % (i,val), end="")
print("\n")    






