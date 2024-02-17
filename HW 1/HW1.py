import numpy as np
import subprocess
import time
import os
import sys
from scipy.stats import multivariate_normal
from scipy.integrate import quad, nquad, simpson
from datetime import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd



# Definitions for Q2A1 error integral, including variations for benchmarking computation time....
def fn_integrand(x,Mu,C0,C1,Pr):
      PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(x) * Pr[0]
      PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C1).pdf(x) * Pr[1]
      return np.minimum(PDF_x_0, PDF_x_1)

def fn_integrand_1d_Gauss(x0,Mu,C0,C1,Pr):
      d=1
      PDF_explicit_0 =  1 / np.sqrt((2*math.pi)**d * np.linalg.det(C0)) * math.exp(-1/2 * (x0-Mu[0])*np.linalg.inv(C0)*(x0-Mu[0])) * Pr[0]
      PDF_explicit_1 =  1 / np.sqrt((2*math.pi)**d * np.linalg.det(C1)) * math.exp(-1/2 * (x0-Mu[1])*np.linalg.inv(C1)*(x0-Mu[1])) * Pr[1]
      return np.minimum(PDF_explicit_0, PDF_explicit_1)

def fn_integrand_2d_Gauss(x0,x1,Mu,C0,C1,Pr):
      d=2
      x = np.array([x0, x1])
      PDF_explicit_0 =  1 / np.sqrt((2*math.pi)**d * np.linalg.det(C0)) * math.exp(-1/2 * ((x-Mu[0,:]).T)@np.linalg.inv(C0)@(x-Mu[0,:])) * Pr[0]
      PDF_explicit_1 =  1 / np.sqrt((2*math.pi)**d * np.linalg.det(C1)) * math.exp(-1/2 * ((x-Mu[1,:]))@np.linalg.inv(C1)@(x-Mu[1,:])) * Pr[1]
      return np.minimum(PDF_explicit_0, PDF_explicit_1)

def fn_integrand_2d_Gauss_DenomAndInv(x0,x1,Mu,Factor0,Factor1,C0inv,C1inv,Pr):
      d=2
      x = np.array([x0, x1])
      PDF_explicit_0 =  Factor0 * math.exp(-1/2 * ((x-Mu[0,:]).T)@C0inv@(x-Mu[0,:])) * Pr[0]
      PDF_explicit_1 =  Factor1  * math.exp(-1/2 * ((x-Mu[1,:]))@C1inv@(x-Mu[1,:])) * Pr[1]
      return np.minimum(PDF_explicit_0, PDF_explicit_1)

def fn_integrand_3d_Gauss_DenomAndInv(x0,x1,x2,Mu,Factor0,Factor1,C0inv,C1inv,Pr):
      d=3
      x = np.array([x0, x1, x2])
      PDF_explicit_0 =  Factor0 * math.exp(-1/2 * ((x-Mu[0,:]).T)@C0inv@(x-Mu[0,:])) * Pr[0]
      PDF_explicit_1 =  Factor1  * math.exp(-1/2 * ((x-Mu[1,:]))@C1inv@(x-Mu[1,:])) * Pr[1]
      return np.minimum(PDF_explicit_0, PDF_explicit_1)

def fn_integrand_4d_Gauss_DenomAndInv(x0,x1,x2,x3,Mu,Factor0,Factor1,C0inv,C1inv,Pr):
      d=4
      x = np.array([x0, x1, x2, x3])
      PDF_explicit_0 =  Factor0 * math.exp(-1/2 * ((x-Mu[0,:]).T)@C0inv@(x-Mu[0,:])) * Pr[0]
      PDF_explicit_1 =  Factor1  * math.exp(-1/2 * ((x-Mu[1,:]))@C1inv@(x-Mu[1,:])) * Pr[1]
      return np.minimum(PDF_explicit_0, PDF_explicit_1)

def fn_integrand_3d_mvn(x0,x1,x2,Mu,C0,C1,Pr):
      x = np.array([x0,x1,x2])
      PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(x) * Pr[0]
      PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C1).pdf(x) * Pr[1]
      return np.minimum(PDF_x_0, PDF_x_1)

def fn_integrand2(x0,x1,x2,x3,Mu,C0,C1,Pr,T0):
      x = np.array([x0,x1,x2,x3])
      PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(x) * Pr[0]
      PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C1).pdf(x) * Pr[1]
      #print(f'Input: {x}, {Mu}, {C0}, {C1}, {Pr}. Output: {np.minimum(PDF_x_0, PDF_x_1)}')
      T1 = datetime.now()
      print(f'T Elapsed: {np.round((T1-T0).total_seconds(),1)}. Input: {x}, Output:{np.minimum(PDF_x_0, PDF_x_1)}')
      return np.minimum(PDF_x_0, PDF_x_1)



# Simulated Dataset Distribution
Pr = np.array([0.35, 0.65])
Mu = np.array([[-1, -1, -1, -1],
      [1, 1, 1, 1]])
C0 = np.array([
    [5, 3, 1, -1],
    [3, 5, -2, -2],
    [1, -2, 6, 3],
    [-1, -2, 3, 4]])
C1 = np.array([
    [1.6, -0.5, -1.5, -1.2],
    [-0.5, 8, 6, -1.7],
    [-1.5, 6, 6, 0],
    [-1.2, -1.7, 0, 1.8]])

# Generate samples
N = 10000
def Generate_Save_Dataset(C0,C1,Mu,N):
      Class0Samp = np.random.multivariate_normal(Mu[0,:],C0,N)
      Class1Samp = np.random.multivariate_normal(Mu[1,:],C1,N)
      Class0Samp = np.hstack((Class0Samp,np.zeros((N,1)))) # Assign labels
      Class1Samp = np.hstack((Class1Samp,np.ones((N,1))))
      Dataset = np.vstack((Class0Samp[:int(Pr[0] * N),:],Class1Samp[:int(Pr[1] * N),:]))
      np.random.shuffle(Dataset)
      np.save('Dataset_1e5.npy',Dataset)
      return Dataset


# Q2A1.
# Computing p(error)
def Compute_Theor_Error(C0,C1,Mu,Pr):
      lims_4d = [[-np.inf, np.inf] for _ in range(4)]
      # lims_3d = [[-np.inf, np.inf] for _ in range(3)]
      # lims_2d = [[-np.inf, np.inf] for _ in range(2)]
      # lims_1d = [[-np.inf, np.inf]]

      # Performance evaluation. After nquad of a 4-dimensional scipy-mvn-generated integrand took more than 10 hours and then raised an error. 
      # Using zero-mean and unity-variance Gaussians to ensure correct answer
      #Mu = np.zeros((2,4))
      #C0 = np.eye(4)
      #C1 = 1*C0

      dims = 4 # Uncomment the following four lines for verifying the line of the Ultimate Run.
      Factor0 = 1 / np.sqrt((2*math.pi)**dims * np.linalg.det(C0[:dims,:dims])) 
      Factor1 = 1 / np.sqrt((2*math.pi)**dims * np.linalg.det(C1[:dims,:dims])) 
      C0inv = np.linalg.inv(C0[:dims,:dims])
      C1inv = np.linalg.inv(C1[:dims,:dims])
      T0 = datetime.now()
      #result,error = nquad(fn_integrand_1d_Gauss, lims_1d, args=(Mu[:,:dims], C0[:dims,:dims], C1[:dims,:dims], Pr)) # For N(0,1), using scipy-mvn, obviously correct answer, <0.0 s
      #result,error = nquad(fn_integrand_2d_Gauss, lims_2d, args=(Mu[:,:dims], C0[:dims,:dims], C1[:dims,:dims], Pr)) # For N(0,0, 1,1),using scipy-mvn,  also correct answer ~4.3 s
      #result,error = nquad(fn_integrand_2d_Gauss_DenomAndInv, lims_2d, args=(Mu[:,:dims],Factor0,Factor1,C0inv,C1inv,Pr)) # ~0.4 s when supplying Factor and Inverted C
      # Final runs
      #result,error = nquad(fn_integrand_3d_Gauss_DenomAndInv, lims_3d, args=(Mu[:,:],Factor0,Factor1,C0inv,C1inv,Pr), opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-3, 'limit': 100}) # 25 s after supplying these Opts
      #result,error = nquad(fn_integrand_3d_Gauss_DenomAndInv, lims_3d, args=(Mu[:,:],Factor0,Factor1,C0inv,C1inv,Pr), opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-1, 'limit': 20}) # 9 s with the lower subdivision limit and somewhat larger tolerance
      # The ultimate run
      result,error = nquad(fn_integrand_4d_Gauss_DenomAndInv, lims_4d, args=(Mu[:,:dims],Factor0,Factor1,C0inv,C1inv,Pr), opts = {'epsabs': 1.49e-3, 'epsrel': 1.49e-1, 'limit': 20}) 
      print(result)
      T1 = datetime.now()
      print(f'T Elapsed: {np.round((T1-T0).total_seconds(),1)}')
      # Ultimate result: A. 0.01979454056794788,   ~2%   # T Elapsed: 963.0 s.  Less than a minute?...
      # Part C.... Over 30 minutes, aborted.
      # Part D.... Not even started.

# Q2B. Compute Chernoff and Bhattacharyya bounds
def ChBhattBounds(Mu,C0,C1,Pr):
      f_kexp_b = lambda beta, m1, m0, c0, c1: beta*(1-beta) / 2 * (m1-m0).T @ np.linalg.inv(beta*c0 + (1-beta)*c1) @ (m1-m0) + 1/2 * np.log(np.linalg.det(beta*c0 + (1-beta)*c1)/(np.linalg.det(c0))**beta / (np.linalg.det(c1))**(1-beta)) 
      f_p_err = lambda beta, m0, m1, c0, c1, Pr: Pr[0]**beta * Pr[1]**(1-beta) * math.exp(-f_kexp_b(beta,m0,m1,c0,c1))
      Nbeta = 1000
      p_err_Ch = np.full((Nbeta,1),np.nan)
      beta_List = np.linspace(0,1,Nbeta)
      for ib,beta in enumerate(beta_List):
            p_err_Ch[ib] = f_p_err(beta,Mu[0,:],Mu[1,:],C0,C1,Pr)
      ibmin = np.argmin(p_err_Ch)
      beta_minErr = beta_List[ibmin]
      Bhatt_Err = p_err_Ch[int(Nbeta/2)]

      annotation_point = (beta_minErr, min(p_err_Ch))
      annotation_text = f'Error u-bound = {np.round(min(p_err_Ch)[0],3)}\nat b = {np.round(beta_minErr,2)}'

      annotation_point_Bhatt = (0.5, Bhatt_Err)
      annotation_text_Bhatt = f'Bhattacharyya bound = {np.round(Bhatt_Err[0],3)}'

      plt.figure(figsize=(5,5))
      plt.subplot(1,1,1)
      plt.plot(beta_List,p_err_Ch)
      plt.annotate(annotation_text, xy=annotation_point, xytext=(0.3, 0.5*max(p_err_Ch)),
                  arrowprops=dict(facecolor='black', arrowstyle='->'),
                  fontsize=8, ha='center', va='center')
      plt.annotate(annotation_text_Bhatt, xy=annotation_point_Bhatt, xytext=(0.7, 0.8*max(p_err_Ch)),
                  arrowprops=dict(facecolor='black', arrowstyle='->'),
                  fontsize=8, ha='center', va='center')
      plt.xlabel(r'$\beta$')
      plt.ylabel(r'$Chernoff P_{error}$')
      plt.savefig('HW1_Q2B.png')
      plt.savefig('HW1_Q2B.svg')
      plt.show()      
      # plt.hist(exp_quot, bins = int(np.sqrt(N)), color='blue', edgecolor='black')



# Q2A2 and Q2A3, and Q2C,D,E
def FindROC_Err(Pr,Mu,C0,C1,N,Mcost=[]):
      # A straightforward gamma, based on the ratio of priors is 
      gamma0 = Pr[0]/Pr[1] # = 7/13 ~ 0.58
      Dataset = np.load('Dataset_1e5.npy')

      # 2-dimensional risk matrix supported, default cost 1 to an error, cost 0 to correct.
      CostVar = True
      if not(len(Mcost) > 0):
            Mcost = np.array([[0, 1],[1, 0]])
            CostVar = False
      r = (Mcost[1,0] - Mcost[0,0]) / (Mcost[0,1] - Mcost[1,1])


      # Function for the quotient reduced to difference of the exponents
      f_exp_quot = lambda x,m0,m1,c0,c1: (x-m0).T @ np.linalg.inv(c0) @ (x-m0) - (x-m1).T @ np.linalg.inv(c1) @ (x-m1)
      # And for the quotient of the two posteriors
      f_p_quot = lambda x,m0,m1,c0,c1: (np.linalg.det(c0) / np.linalg.det(c1))**(1/2) * math.exp(1/2 * f_exp_quot(x,m0,m1,c0,c1))

      # If implementing a DR with f_exp_quot, factor out the scale from both posterior PDFs, transfer them to the RHS, and take a log of both sides. A single value
      #gamma1 = 2 * np.log(Pr[0]) - 2 * np.log(Pr[1]) + np.log(np.linalg.det(C1)) - np.log(np.linalg.det(C0)) # = -5.544
      # If sliding...
      #f_gamma1 = lambda gamm_slide: 2 * np.log(gamm_slide) + np.log(np.linalg.det(C1)) - np.log(np.linalg.det(C0))
      f_gamma1_fromLog = lambda gamm_log: np.sqrt(np.linalg.det(C0) / np.linalg.det(C1) * math.exp(gamm_log) * r) # Including risk factor here...
      #f_gamma1Log = lambda gamm_slide: 2 * np.log(gamm_slide) + np.log(np.linalg.det(C1)) - np.log(np.linalg.det(C0))


      # Estimate the distribution of the quotient of posteriors to determine range and step of gamma0
      post_quot = np.full((N,1),np.nan)
      for i in range(N):
            post_quot[i] = f_p_quot(Dataset[i,:4],Mu[0,:],Mu[1,:],C0,C1)
      plt.figure(figsize=(4,4))
      plt.hist(post_quot, bins = int(np.sqrt(N)), color='blue', edgecolor='black')
      plt.xlabel('Quotient of posteriors L=1 / L=0')
      plt.ylabel('Counts')
      if len(Mcost) > 0:
            plt.savefig(f'HW1_Q2E_quotientHist_B{Mcost[0,1]}.png')
      else:
            plt.savefig(f'HW1_Q2E_quotientHist_Norisk.png')
      #plt.show()
      plt.close()
      # RESULT: the range of the quotient is up to 40 orders of magnitude. Log scaling suggested.

      # Estimate the distribution of the quotient of the exponents
      exp_quot = np.full((N,1),np.nan)
      for i in range(N):
            exp_quot[i] = f_exp_quot(Dataset[i,:4],Mu[0,:],Mu[1,:],C0,C1)
      plt.figure(figsize=(4,4))
      plt.hist(exp_quot, bins = int(np.sqrt(N)), color='blue', edgecolor='black')
      plt.xlabel('Quotient of exponents L=1 / L=0')
      plt.ylabel('Counts')
      if len(Mcost) > 0:
            plt.savefig(f'HW1_Q2E_exp_diffHist_B{Mcost[0,1]}.png')
      else:
            plt.savefig(f'HW1_Q2E_exp_diffHist_Norisk.png')
      #plt.show()
      plt.close()
      # RESULT: gamma [-600,200] would cover the whole region of the quotient values. Most dense in [-50 to 50]
      # The run with CoarseStep 50 and FineStep 1 revealed that error is <=1.83 at 0.428 < gamm < 1.165, and for -7<gamm_exp<-4



      # Implementing sliding-threshold classifier using log-scale of the gamma_exp, against which a difference of exponents is compared
      TPR = []
      FPR = []
      FNR = []
      TNR = []
      Risk = []
      ErrEmp = []
      gamm = []
      WhileTrue = True


      # StepCoarse = 50
      # StepFine = 1
      # StepFinest = 0.1
      StepsPerRange = 6
      # Set up ranges, based on data percentiles...
      pmin,pmax = min(exp_quot)[0],max(exp_quot)[0]
      p2,p98 = np.percentile(exp_quot,[2.5,97.5])
      p25,p75 = np.percentile(exp_quot,[25,75])
      Ranges = np.array([[pmin, p2],  
                [p2, p25],
                [p25, p75],
                [p75, p98],
                [p98, pmax]])
      
      
      gamm_log_List = np.hstack((np.linspace(Ranges[0,0],Ranges[0,1],StepsPerRange),np.linspace(Ranges[1,0],Ranges[1,1],StepsPerRange*3),
                                 np.linspace(Ranges[2,0],Ranges[2,1],StepsPerRange*5),np.linspace(Ranges[3,0],Ranges[3,1],StepsPerRange*3),
                                 np.linspace(Ranges[4,0],Ranges[4,1],StepsPerRange)))

      #gamm_log_List = np.hstack((np.arange(-600,-50,StepCoarse),np.arange(-50,50,StepFine),np.arange(50,200,StepCoarse)))
      #gamm_log_List = np.hstack((np.arange(-600,-50,StepCoarse),np.arange(-50,-7,StepFine),np.arange(-7,-4,StepFinest),np.arange(-4,50,StepFine),np.arange(50,200,StepCoarse)))

      # LHS in the reduced classifier is always the same for the given dataset, therefore it can be computed single time for all gammas.
      LHS_Data = np.full((len(Dataset),1),np.nan)
      for i in range(N):
            LHS_Data[i] = f_exp_quot(Dataset[i,:4],Mu[0,:],Mu[1,:],C0,C1)


      for gamm_log_slide in gamm_log_List:
            Label_dec = np.zeros((N,1),dtype=int)
            Label_metrics = np.zeros((N,5),dtype=bool) # [Correct label, TP, FP, FN, TN]
            for i in range(N):
                  Label_dec[i] = 1 if (LHS_Data[i] >= gamm_log_slide) else 0

            Label_metrics[:,0] = (Dataset[:,4] == Label_dec.T) # Correct decision
            Label_metrics[:,1] = (Label_dec.T == 1) & Label_metrics[:,0] # TP
            Label_metrics[:,2] = (Label_dec.T == 1) & ~Label_metrics[:,0] # FP
            Label_metrics[:,3] = (Label_dec.T == 0) & ~Label_metrics[:,0] # FN
            Label_metrics[:,4] = (Label_dec.T == 0) & Label_metrics[:,0] # TN
            Pos = np.count_nonzero(Dataset[:,4] == 1)
            Neg = np.count_nonzero(Dataset[:,4] == 0)
            TPR_temp = np.count_nonzero(Label_metrics[:,1]) / Pos
            FPR_temp = np.count_nonzero(Label_metrics[:,2]) / Neg
            FNR_temp = np.count_nonzero(Label_metrics[:,3]) / Pos
            TNR_temp = np.count_nonzero(Label_metrics[:,4]) / Neg
            Risk_temp = - Mcost[0,0] * TNR_temp - Mcost[1,1] * TPR_temp + Mcost[0,1] * FNR_temp * Pos/N + Mcost[1,0] * FPR_temp * Neg/N # When no Mcost supplied, this becomes Error.
            Risk.append(Risk_temp)
            gammTemp = f_gamma1_fromLog(gamm_log_slide)
            gamm.append(gammTemp)
            TPR.append(TPR_temp)
            FPR.append(FPR_temp)
            FNR.append(FNR_temp)
            TNR.append(TNR_temp)
            ErrEmp.append(np.count_nonzero(~Label_metrics[:,0]) / N * 100)
            print(f'ln(gamm_exp) = {gamm_log_slide}, gamm_PDF = {gammTemp}:  TPR:{round(TPR_temp,3)}, FPR:{round(FPR_temp,3)}, Err:{round(ErrEmp[-1],3)}%')
      

      # Q2A3
      # Minimum error at lambda...
      iMinErr = np.argmin(ErrEmp)
      gamm_Min = gamm[iMinErr]
      print(f'Minimum error {ErrEmp[iMinErr]}% attained at gamma = {gamm_Min}')
      annotation_point = (FPR[iMinErr], TPR[iMinErr])
      annotation_text = \
            f'Min Error = {np.round(ErrEmp[iMinErr],2)}%\n\
            Gamma = {np.round(gamm_Min,2)}\n\
            FPR = {np.round(FPR[iMinErr]*100,2)}%\n\
            TPR = {np.round(TPR[iMinErr]*100,2)}%\n\
            FNR = {np.round(FNR[iMinErr]*100,2)}%\n\
            TNR = {np.round(TNR[iMinErr]*100,2)}%\n\
            Risk = {np.round(Risk[iMinErr],4)}a.u.\n\
            B = {np.round(Mcost[0,1],2)}'
                        
      # Plot ROC
      fsize = (8,4)
      Ncols = 2
      if CostVar:
            fsize = (13,4)
            Ncols = 3

      plt.figure(figsize=fsize)
      plt.subplot(1,Ncols,1)
      plt.plot(FPR,TPR)
      plt.xlabel('FPR')
      plt.ylabel('TPR')
      plt.annotate(annotation_text, xy=annotation_point, xytext=(annotation_point[0] + 0.35, 0.3),
                  arrowprops=dict(facecolor='black', arrowstyle='->'),
                  fontsize=10, ha='left', va='center')
      # And error
      plt.subplot(1,Ncols,2)
      plt.plot(gamm,ErrEmp)
      plt.xscale('log')
      plt.xlabel(r'$\gamma$')
      plt.ylabel('Error (%)')
      
      # And risk, if different from Error...
      if CostVar:
            plt.subplot(1,Ncols,3)
            plt.plot(gamm,Risk)
            plt.xscale('log')
            plt.xlabel(r'$\gamma$')
            plt.ylabel('Risk')

      plt.tight_layout()
      if CostVar:
            plt.savefig(f'HW1_Q2E_1e5_B{Mcost[0,1]}.png')
            plt.savefig(f'HW1_Q2E_1e5_B{Mcost[0,1]}.svg')
      else:
            plt.savefig(f'HW1_Q2E_1e5_Norisk.png')
            plt.savefig(f'HW1_Q2E_1e5_Norisk.svg')
      
      #plt.show()
      plt.close()

      # Compute AUC and return stuff
      ROC_AUC = -simpson(TPR,FPR)
      print(ROC_AUC)
      return gamm[iMinErr], ErrEmp[iMinErr], FPR[iMinErr], TPR[iMinErr], Risk[iMinErr], ROC_AUC



# Q2E
def EvalClassifierWithRisk():
      #Compute_Theor_Error(C0,C1,Mu,Pr)
      BList = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.1, 1.5, 2, 5, 10, 20, 50, 100]
      gamm = np.full((len(BList),1),np.nan)
      ErrEmp = np.full((len(BList),1),np.nan)
      FPR = np.full((len(BList),1),np.nan)
      TPR = np.full((len(BList),1),np.nan)
      Risk = np.full((len(BList),1),np.nan)
      AUC = np.full((len(BList),1),np.nan)
      for iB,B in enumerate(BList):
            Mcost = np.array([
                  [0, B],
                  [1, 0]
            ])
            r = 1
            if len(Mcost) > 0:
                  r = (Mcost[1,0] - Mcost[0,0]) / (Mcost[0,1] - Mcost[1,1])
            gamm[iB], ErrEmp[iB], FPR[iB], TPR[iB], Risk[iB], AUC[iB] = FindROC_Err(Pr,Mu,C0,C1,N,Mcost) 
            print(f'B = {B}, gamm_min = {gamm[iB]}, Err_min = {np.round(ErrEmp[iB],2)}%,  TPR:{np.round(TPR[iB] * 100,2)}%, FPR:{np.round(FPR[iB],2)}%, Risk:{np.round(Risk[iB],3)}')

      plt.figure(figsize=(12,6))
      plt.subplot(2,3,1)
      plt.plot(BList,ErrEmp)
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('Min Empirical Error')

      plt.subplot(2,3,4)
      plt.plot(BList,Risk)
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('Min Empirical Risk')

      plt.subplot(2,3,2)
      plt.plot(BList,gamm)
      plt.xscale('log')
      plt.yscale('log')
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('Gamma')

      plt.subplot(2,3,5)
      plt.plot(BList,AUC)
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('AUC')

      plt.subplot(2,3,3)
      plt.plot(BList,FPR)
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('FPR')

      plt.subplot(2,3,6)
      plt.plot(BList,TPR)
      plt.xlabel('B (cost of 0|1 w.r.t. cost of 1|0)')
      plt.ylabel('TPR')

      plt.tight_layout()
      plt.savefig(f'HW1_Q2E_1e5_Final.png')
      plt.savefig(f'HW1_Q2E_1e5_Final.svg')

      plt.show()
      plt.close()








# ####Q2A1.
# Generate_Save_Dataset(C0,C1,Mu,N)
# Compute_Theor_Error(C0,C1,Mu,Pr)

# ####Q2B. Compute Chernoff and Bhattacharyya bounds
# ChBhattBounds(Mu,C0,C1,Pr)

# ####Q2C
# #Assume isotropic distributions.
# for i in range(4):
#     for j in range(4):
#         if i != j:
#             C0[i, j] = 0
#             C1[i, j] = 0
# #Compute_Theor_Error(C0,C1,Mu,Pr)
# FindROC_Err(Pr,Mu,C0,C1,N)

# ####Q2D
# #Assume same covariances, estimate them from the dataset
# Dataset = np.load('Dataset_1e5.npy')
# Dataset[Dataset[:,4]==0,:4] -= Mu[0,:]
# Dataset[Dataset[:,4]==1,:4] -= Mu[1,:]
# C = np.cov(Dataset[:,:4],rowvar=False)
# #Compute_Theor_Error(C0,C1,Mu,Pr)
# FindROC_Err(Pr,Mu,C,C,N)

# ####Q2E
# #Vary risk by changing cost of 0|1 relative to 1|0
# EvalClassifierWithRisk()








# ####Q3 Wine
def Q3_Wine():
      from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
      import pandas as pd

      # Using continuous (0/1) representation for red/white wines
      df = pd.read_csv('Q3_ExternalDatasets\\winequality-red.csv', delimiter=';')
      df['type'] = 0
      df = df[['type'] + df.columns[:-1].tolist()]
      df1 = pd.read_csv('Q3_ExternalDatasets\\winequality-white.csv', delimiter=';')
      df1['type'] = 1
      df1 = df1[['type'] + df1.columns[:-1].tolist()]
      df = pd.concat((df,df1))
      Feats = df.columns[:-1].tolist()
      print('Features: ', Feats)
      # For compatibility with the previous code syntax
      Dataset = df.to_numpy()
      ClassTrue = Dataset[:,-1]
      ClassTrue = ClassTrue.astype(int)
      Dataset = Dataset[:,:-1]
      # Estimating class-conditional priors, means, and covariances, assuming Gaussians.
      NData = len(Dataset)
      NClass = 11
      Pr = np.full((NClass,1),np.nan)
      Mu = np.full((NClass,len(Feats)),np.nan)
      C = np.full((NClass,len(Feats),len(Feats)),np.nan)
      rvs = np.empty(NClass, dtype=object)
      CondNum = np.full((NClass,1),np.nan)
      CondNumUpd = np.full((NClass,1),np.nan)
      PostPDF = np.zeros((len(Dataset),NClass),dtype=float)
      PostPDF_Proper = np.zeros((len(Dataset),NClass),dtype=float)

      SingCondNumThresh = 1e4
      for ic in range(11):
            X = Dataset[ClassTrue == ic,:]
            Pr[ic] = len(X) / NData
            if len(X) < 1:
                  continue
            Mu[ic] = np.nanmean(X,axis=0)
            Xc = X - Mu[ic]
            Ctemp = np.cov(X,rowvar=False)
            C[ic,:,:]  = Ctemp
            # Regularize? Find condition number as max(abs(eig))/min(abs(eig))
            Eigs = np.linalg.eigvals(Ctemp)
            CondNum[ic] = max(abs(Eigs)) / min(abs(Eigs))

            # Regularize if any CondNum > ...? > 1e4 Found empirically to maximize accuracy
            CondNumUpd[ic] = CondNum[ic]
            while CondNumUpd[ic] > SingCondNumThresh:
                  L = 0.01 * np.trace(Ctemp) / np.linalg.matrix_rank(Ctemp)
                  Ctemp = Ctemp + L * np.eye(len(Feats))
                  C[ic,:,:] = Ctemp
                  Eigs = np.linalg.eigvals(Ctemp)
                  CondNumUpd[ic] = max(abs(Eigs)) / min(abs(Eigs))

            # Find PDFs and posteriors given the dataset
            rvs[ic] = (multivariate_normal(mean=Mu[ic], cov=Ctemp))
            PostPDF[:,ic] = rvs[ic].pdf(Dataset[:,:]) * Pr[ic]
            

      # Find P(x) for each row and normalize -- should not affect classification.
      for i in range(NData):
            PostPDF_Proper[i,:] = PostPDF[i,:] / np.sum(PostPDF[i,:])

      # Apply classification as argmax(Post_PDF_Proper)
      ClassPred = np.argmax(PostPDF_Proper, axis=1).astype(int)
      # ClassPred = ClassPred + 1 # First thought that quality is [1..10] not [0..10]. Strange, but this line may increase accuracy, although still lower than sklearn.GaussianNB
      # Evaluate performance, make confusion matrix
      ConfMat = confusion_matrix(ClassTrue,ClassPred)
      class_labels = sorted(set(ClassTrue) | set(ClassPred))
      conf_matrix_df = pd.DataFrame(ConfMat, index=class_labels, columns=class_labels)
      print(conf_matrix_df)

      Errors = np.count_nonzero(ClassTrue != ClassPred)
      accuracy = accuracy_score(ClassTrue, ClassPred)
      precision = precision_score(ClassTrue, ClassPred, average=None)
      sensrecall = recall_score(ClassTrue, ClassPred, average=None)
      print(f'Accuracy: {np.round(accuracy*100,2)}%, Error: {np.round(Errors/NData*100,2)}% Precision: {np.round(precision*100,2)}%, Recall: {np.round(sensrecall*100,2)}%')

      # Verify that manual calculation...
      print('Compare with sklearn-GaussianNB....')
      from sklearn.model_selection import train_test_split
      from sklearn.naive_bayes import GaussianNB
      from sklearn import metrics
      gnb = GaussianNB()
      gnb.fit(Dataset, ClassTrue)
      # Make predictions on the test set
      y_pred = gnb.predict(Dataset)
      conf_matrix_df = pd.DataFrame(confusion_matrix(ClassTrue,y_pred), index=class_labels, columns=class_labels)
      print(conf_matrix_df)
      # Evaluate the classifier
      accuracy = metrics.accuracy_score(ClassTrue, y_pred)
      print("Accuracy:", accuracy)  # 56.5% but what does that mean?...


      # Visualize the dataset via pairplots
      # import seaborn as sns
      # g = sns.pairplot(df, diag_kind='kde', kind="hist", hue="quality", corner=True,height=0.5)
      # g.map_lower(sns.kdeplot, levels=4, color=".2")
      # label_font_params = {'fontsize': 8, 'fontweight': 'bold'}
      # # Apply the font parameters to the labels in the pairplot
      # #g.set(fontscale=2.5)
      # #g.set_label_params(label=label_font_params, which='both')
      # plt.savefig('HW1_Q3_WineRed_PairPlot.png')
      # plt.show()
      # plt.close()


      # Visualize the dataset via 3 PCS
      # NPCs = 3
      # Dataset_norm = (Dataset - np.mean(Dataset, axis=0)) / np.std(Dataset, axis=0, ddof=1)
      # cov_Data = np.cov(Dataset_norm, rowvar=False) # regularize?
      # eigenvalues, eigenvectors = np.linalg.eigh(cov_Data)
      # # sort eigenvectors by eigenvals descending
      # sorted_indices = np.argsort(eigenvalues)[::-1]
      # eigenvalues = eigenvalues[sorted_indices]
      # eigenvectors = eigenvectors[:, sorted_indices]
      # PCs = eigenvectors[:, :NPCs]
      # # You can project your data onto the selected principal components
      # PCsProj = np.dot(Dataset_norm, PCs)


      # fig = plt.figure(figsize=(8,8))
      # ax = fig.add_subplot(111, projection='3d')
      # axsc = ax.scatter(PCsProj[:, 0], PCsProj[:, 1], PCsProj[:, 2], marker='o', c=ClassTrue, cmap='viridis', label='Quality', alpha=0.6)
      # cbar = plt.colorbar(axsc, ticks=np.unique(ClassTrue), orientation='vertical')
      # scatter_legend = ax.scatter([], [], [], c=[], cmap='viridis', label='Quality')
      # cbar.set_label('Color Feature')
      # ax.set_xlabel('PC1')
      # ax.set_ylabel('PC2')
      # ax.set_zlabel('PC3')
      # plt.title('3D Scatter Plot of Data Projected onto the First 3 PCs')
      # plt.savefig('HW1_Q3_WineRed_PCs.png')
      # plt.show()
      # plt.close()







# Q3 run for the wine dataset (allow plots, but seaborn pairplot takes long!)
#Q3_Wine()


print('Hello Worlds and end of the script...')
#print(sum(np.arange(0,101)))

