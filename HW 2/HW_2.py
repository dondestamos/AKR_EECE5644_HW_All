import numpy as np
import os
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# Categorical NB implementation, allowing externally-defined test/train ratio, smoothing parameter, randomizer seed, and expected number of categories per feature.
def CategoricalClass_VarSmooth_VarSplit(X,y,catsperfeature,a_train,seed,var_alpha):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a_train, random_state=seed)
    #print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

    #print("Training classifier with smoothing alpha %d..." % (var_alpha), end="")
    classifier = CategoricalNB(alpha=var_alpha,min_categories = catsperfeature)
    classifier.fit(X_train, y_train)

    #get preditions
    y_pred =  classifier.predict(X_test) 
    acc = accuracy_score(y_test,y_pred)   #ACC = (TP + TN) / (P+N)
    f1 = f1_score(y_test,y_pred)    #F1 = 2 * TP / (2 * TP + FN + FP)
    cm = confusion_matrix(y_test,y_pred) # 0 poisonous/unknown, 1 edible. Item [0,1] is a poisonous mushroom classified as edible.......
    cm_most_risk = cm[0,1] / len(y_pred) # False-positives appear extremely important in this problem
    probs = classifier.predict_proba(X_test) # discriminant function values per class
    fpr,tpr,thresholds = roc_curve(y_test,probs[:,1])
    auc = roc_auc_score(y_test,probs[:,1])

    print(f"Alpha: {var_alpha}, Accuracy: {np.round(acc*100,2)}%, F1: {np.round(f1*100,2)}%, FPs: {np.round(cm_most_risk*100,2)}%, AUC: {np.round(auc*100,2)}%")
    return acc,f1,cm_most_risk,auc

# Multinomial NB implementation, allowing externally-defined test/train ratio, smoothing parameter, and randomizer seed.
def MultinomialClass_VarSmooth_VarSplit(X,y,a_train,seed,var_alpha):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a_train, random_state=seed)
    #print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

    classifier = MultinomialNB(alpha=var_alpha)
    classifier.fit(X_train, y_train.values.ravel())

    #get preditions
    y_pred =  classifier.predict(X_test) 
    acc = accuracy_score(y_test,y_pred)

    print(f"Alpha: {var_alpha}, Accuracy: {np.round(acc*100,2)}%, F1: {None}, FPs: {None}, AUC: {None}")
    return acc,None,None,None





# ####Q3 Mushrooms
def Q3_Mushrooms():
    X = np.genfromtxt('X_msrm.csv', delimiter=',',dtype=int)
    y = np.genfromtxt('y_msrm.csv', delimiter=',',dtype=int)
    #balanced?
    print(f'Class instances: {np.count_nonzero(y==0)} poisonous/unknown and {np.count_nonzero(y==1)} edible')
    # categories per feature
    Feat_Cats = []
    for icol in range(X.shape[1]):
        Feat_Cats.append(len(np.unique(X[:,icol])))


    splitseed = None # 42 or any constant for consistent rnd.
    a_train = 0.1
    var_alpha = 1e-5
    Nalpha = 100
    var_alphaListLin = np.linspace(2**(-15),2**5,Nalpha) 
    var_alphaListLog = np.exp(np.linspace(-15*np.log(2),5*np.log(2),Nalpha))
    Rows = []

    alphaList = var_alphaListLog
    for var_alpha in alphaList:
        acc,f1,cm_most_risk,auc = CategoricalClass_VarSmooth_VarSplit(X,y,Feat_Cats,a_train,splitseed,var_alpha)
        new_row = {"Alpha": var_alpha, "Acc": acc, "F1": f1, "FPs": cm_most_risk, 'AUC':auc} 
        Rows.append(new_row)
    ClassRes = pd.DataFrame(Rows)

    iMaxAUC = np.argmax(ClassRes['AUC'])
    alpha_Max = ClassRes.loc[iMaxAUC,'Alpha']
    print(f'Max AUC {ClassRes.loc[iMaxAUC,'AUC']} attained at alpha = {alpha_Max}')

    fig = plt.figure(figsize=(12,3))
    axA = fig.add_subplot(1,5,1)
    axA.plot(np.arange(0,Nalpha),ClassRes['Alpha'])
    axA.set_xlabel('Iter')
    axA.set_ylabel('Alpha')

    axAcc = fig.add_subplot(1,5,2)
    axAcc.plot(ClassRes['Alpha'],ClassRes['Acc'])
    axAcc.set_xlabel('Alpha')
    plt.xscale('log')
    axAcc.set_ylabel('Accuracy')

    axF1 = fig.add_subplot(1,5,3)
    axF1.plot(ClassRes['Alpha'],ClassRes['F1'])
    axF1.set_xlabel('Alpha')
    plt.xscale('log')
    axF1.set_ylabel('F1')

    axFPs = fig.add_subplot(1,5,4)
    axFPs.plot(ClassRes['Alpha'],ClassRes['FPs'])
    axFPs.set_xlabel('Alpha')
    plt.xscale('log')
    axFPs.set_ylabel('FP (false edibles)')

    axAuc = fig.add_subplot(1,5,5)
    axAuc.plot(ClassRes['Alpha'],ClassRes['AUC'])
    axAuc.set_xlabel('Alpha')
    plt.xscale('log')
    axAuc.set_ylabel('AUC')

    plt.tight_layout()
    #plt.show()
    #fig.savefig("Q3_AlphaLog_1_1000_Train10.png", bbox_inches="tight")
    plt.close







# Obtaining vocabulary and categories for finding the highest-posterior words.
def to_lower_case(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'. 
    """
    return s.lower()

def strip_non_alpha(s):
    """ Remove non-alphabetic characters from the beginning and end of a string. 

    E.g. ',1what?!"' should become 'what'. Non-alphabetic characters in the middle 
    of the string should not be removed. E.g. "haven't" should remain unaltered."""

    s = s.strip()
    if len(s)==0:
        return s
    if not s[0].isalpha():    
        return strip_non_alpha(s[1:])         
    elif not s[-1].isalpha():       
        return strip_non_alpha(s[:-1])        
    else:
        return s

def clean(s):
    """ Create a "clean" version of a string 
    """
    return to_lower_case(strip_non_alpha(s))


def GetVocabulary():
    directory = 'SentenceCorpus/labeled_articles/'
    categories = {}
    vocabulary = {}
    num_files = 0
    for filename in [x for x in os.listdir(directory) if ".txt" in x]:
        num_files +=1
        print("Processing",filename,"...",end="")
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            with open(f,'r') as  fp:
                for line in fp:
                    line = line.strip()
                    if "###" in line:
                        continue
                    if "--" in line:
                        label, words = line.split("--")
                        words = [clean(word) for word in words.split()]
                    else:
                        words = line.split()
                        label = words[0]
                        words = [clean(word) for word in words[1:]]

                    if label not in categories:
                        index = len(categories)
                        categories[label] = index
                        
                    for word in words:
                        if word not in vocabulary:
                            index = len(vocabulary)
                            vocabulary[word] = index
        print(" done") 
    return vocabulary


def Q4_Sentences():
    Xp = pd.read_csv('X_snts.csv',dtype=int) # large csv, better pandas.
    yp = pd.read_csv('y_snts.csv',dtype=int)
    X = Xp
    y = yp
    Feat_Cats = []
    S = []
    
    splitseed = None
    a_train = 0.8
    var_alpha = 1e-5
    Nalpha = 100
    Niteraver = 10
    AccAll = np.full((Nalpha,Niteraver),np.nan)
    var_alphaListLin = np.linspace(2**(-15),2**5,Nalpha)
    var_alphaListLog = np.exp(np.linspace(-15*np.log(2),5*np.log(2),Nalpha))
    ClassParams = []
    alphaList = var_alphaListLog

    for iaver in range(Niteraver):
        Acc = []
        for var_alpha in alphaList:
            acc,f1,cm_most_risk,auc = MultinomialClass_VarSmooth_VarSplit(X,y,a_train,splitseed,var_alpha)
            Acc.append(acc)

        iMaxAcc = np.argmax(Acc)
        alpha_Max = alphaList[iMaxAcc]
        S.append(f'Max Accuracy {Acc[iMaxAcc]} attained at alpha = {alpha_Max}')
        AccAll[:,iaver] = Acc

    for iaver in range(10):
        print(S[iaver])

    fig = plt.figure(figsize=(12,3))
    axA = fig.add_subplot(1,5,1)
    axA.plot(np.arange(0,Nalpha),alphaList)
    axA.set_xlabel('Iter')
    axA.set_ylabel('Alpha')

    axAcc = fig.add_subplot(1,5,2)
    M = np.mean(AccAll,axis=1)
    SD = np.std(AccAll,axis=1)
    plt.fill_between(alphaList, M - SD, M + SD, alpha=0.3)
    axAcc.plot(alphaList,M)
    axAcc.set_xlabel('Alpha')
    plt.xscale('log')
    axAcc.set_ylabel('Accuracy')

    iMaxAcc = np.argmax(M)
    alpha_Max = alphaList[iMaxAcc]
    print(f'Average: max Accuracy {M[iMaxAcc]} (SD {SD[iMaxAcc]}) attained at alpha = {alpha_Max}')

    plt.tight_layout()
    #plt.show()
    fig.savefig("Q4_AlphaLog_1_100_Train80.png", bbox_inches="tight")
    plt.close


def Q4_FindSalientWords(alpha_Max):
    
    Xp = pd.read_csv('X_snts.csv',dtype=int) # large csv, better pandas.
    yp = pd.read_csv('y_snts.csv',dtype=int)
    X = Xp
    y = yp
    Feat_Cats = []
    S = []

    # Use the alpha to continue...

    Vocabulary = GetVocabulary()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=alpha_Max, random_state=None)
    classifier = MultinomialNB(alpha=alpha_Max) #,min_categories = catsperfeature
    classifier.fit(X_train, y_train.values.ravel())

    cats = {'MISC': 0, 'AIMX': 1, 'OWNX': 2, 'CONT': 3, 'BASE': 4} # From process_sentences
    log_p = classifier.feature_log_prob_
    iclass = 0
    for classCat in cats:
        itop = np.argsort(log_p[iclass,:])[-5:]
        Words = [word for word, index in Vocabulary.items() if index in itop]
        print(f'Five words of class {classCat} with the highers posterior probability:')
        print(Words)
        print(log_p[iclass,itop])
        iclass += 1


Q3_Mushrooms()

Q4_Sentences()
# Average: max Accuracy 0.7309265944645006 (SD 0.01147492665554326) attained at alpha = 0.9655982385900086
Q4_FindSalientWords(0.965598239)



print('Done. Inspect?')

