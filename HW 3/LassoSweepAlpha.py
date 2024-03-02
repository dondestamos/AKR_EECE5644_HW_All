import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix
from liftDataset import lift_Dataset

# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")
X = lift_Dataset(X)

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))



#alpha = 0.1

cv = KFold(
        n_splits=5, 
        random_state=42,
        shuffle=True
        )

Nalpha = 500
alphaExpList = np.linspace(-20, 20 + 20/Nalpha, Nalpha)
alphaList = 2 ** (alphaExpList)
amean = []
astd = []
Coefs = np.zeros((Nalpha,X_train.shape[1]))
for ia, alpha in enumerate(alphaList):
    model = Lasso(alpha = alpha)
    scores = cross_val_score(
            model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")
    amean.append(-np.mean(scores))
    astd.append(np.std(scores))
    print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,-np.mean(scores),np.std(scores)) )
    model.fit(X_train, y_train)
    Coefs[ia,:] = model.coef_

iopt = np.argmin(amean)
print("Cross-validation RMSE is minimized for exponent %d, α=%f : %f ± %f" % (alphaExpList[iopt], alphaList[iopt],amean[iopt],astd[iopt]) )


fig = plt.figure(figsize=(8,5))
axRt = fig.add_subplot(1,2,1)
#axRt.plot(alphaList,amean)
axRt.errorbar(alphaList,amean,yerr = astd, fmt='o', markersize=2, color='black', linestyle='none',capsize=0)
plt.xscale('log')
axRt.set_xlabel('Lasso factor')
axRt.set_ylabel('CV RMSE of training')

axC = fig.add_subplot(1,2,2)
axC.plot(alphaList,Coefs)
plt.xscale('log')
axC.set_xlabel('Lasso factor')
axC.set_ylabel('Coefficients')

#plt.tight_layout()
plt.show()
fig.savefig("Q4_N_1000_d_40_Sigma_0.01_Lift_Lasso.png", bbox_inches="tight") # Adjust the name per assignment...
#plt.close


aopt = alphaList[iopt]
model = Lasso(alpha = aopt)
print("Fitting linear model over entire training set...",end="")
model.fit(X_train, y_train)
print(" done")


# Compute RMSE
rmse_train = rmse(y_train,model.predict(X_train))
rmse_test = rmse(y_test,model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))

print("Model parameters:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    if abs(val) > 1e-3:
        print(", β%d: %3.5f" % (i,val), end="")
print("\n")  









