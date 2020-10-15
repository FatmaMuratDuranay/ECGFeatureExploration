
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
import numpy as np
import pandas as pd 
import warnings
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
warnings.simplefilter('ignore')

def siniflar(X,Y):
            numfold=5;
            clf1 = KNeighborsClassifier(n_neighbors=3)
            clf2 = SVC(C=7)
            clf3 = RandomForestClassifier(random_state=23)
            clf4 = GaussianNB()
            clf5 = LinearDiscriminantAnalysis()
            clf6 = Pipeline((
                    ("poly_features", PolynomialFeatures(degree=3)),
                    ("svm_clf", LinearSVC(C=5, loss="hinge"))
               ))
            clf7 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
            clf8 = AdaBoostClassifier(n_estimators=100)
            clf9 = DecisionTreeClassifier()
            clf10 = MLPClassifier(alpha=1, max_iter=100)
            clf11 = QuadraticDiscriminantAnalysis()
            clf12 = SVC(kernel="linear", C=0.025)
            clf13 = SVC(gamma=2, C=1)
            clf14 = LogisticRegression()
            
            staclf = StackingClassifier(classifiers=[ clf2, clf3, clf8, clf9], 
                                      meta_classifier= clf7)
            eclf = VotingClassifier(
                estimators=[('lr', clf2), ('rf', clf3), ('gnb', staclf)],
                voting='hard')
            #clf=clf3;
            #label='Random Forest'
            for clf, label in zip([clf1, clf2, clf3, clf4, clf7, clf8, clf9, clf10, clf11], 
                                  ['KNN', 
                                    'SVC',
                                    'Random Forest', 
                                    'Naive Bayes',
                                    'GradientBoostingClassifier',
                                    'AdaBoostClassifier',
                                    'DecisionTreeClassifier',
                                    'Neural Net',
                                    'QDA']):
                    skfolds = StratifiedKFold(n_splits=numfold, random_state=42)
                    say=0;
                    Acc=np.zeros(numfold)
                    f=open('Confmat.txt','a')
                    f.write(label)
                    #np.savetxt(f, label, fmt="%s",newline=", ")
                    f.write("\n")
                    for train_index, test_index in skfolds.split(X, Y):
                                    clone_clf = clone(clf)
                                    X_train_folds = X[train_index]
                                    y_train_folds = (Y[train_index])
                                    X_test_fold = X[test_index]
                                    y_test_fold = (Y[test_index])
                                    clone_clf.fit(X_train_folds, y_train_folds)
                                    y_pred = clone_clf.predict(X_test_fold)
                                    n_correct = sum(y_pred == y_test_fold)
                                    cmat=confusion_matrix(y_test_fold, y_pred)
                                    Acc[say]=(n_correct / len(y_pred));
                                    np.savetxt(f, cmat, fmt='%1.0f', newline=", ")
                                    f.write("\n")
                                    np.savetxt(f, Acc, fmt='%1.3f', newline=", ")
                                    f.write("\n")
                                    say=say+1;
                    f.close()
                    print("[%s]-->%0.4f " %(label, Acc.mean())) # prints 0.9502, 0.96565 and 0.96495
            
