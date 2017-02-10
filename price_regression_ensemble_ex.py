import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_data():
	usecols = ['Cusip', 'Yield', 'Name', 'State', 'IssueSize', 'IssueType', 'TradeType', 'IssueSource', 'Coupon', 'Maturity', 'RTG_Moody', 'RTG_SP', 'Price']
	#read data
	filename = "./data/TMC_020617.csv"
	d = pd.read_csv(filename, usecols = usecols)
	#make column names all lower case
	d.columns = [x.lower() for x in d.columns]
	
	#merge RTG_Moody and RTG_SP columns based on missing values
	moody_to_sp_dict = {'Aaa':'AAA', 'Aa1':'AA+', 'Aa2':'AA',
		'Aa3':'AA-', 'A1':'A+', 'A2':'A', 'A3':'A-',
		'Baa1':'BBB+', 'Baa2':'BBB', 'Baa3':'BBB-',
		'Ba1':'BB+', 'Ba2':'BB', 'Ba3':'BB-', 'B1':'B+',
		'B2':'B', 'B3':'B-', 'Caa1':'CCC+', 'Caa2':'CCC',
		'Caa3':'CCC-', 'Ca':'CC', 'C':'C', '/':'D'}

	f = lambda x: moody_to_sp_dict[x]

	#correct a typo in rating data entry
	d.loc[d.rtg_moody == 'Ca1', 'rtg_moody'] = 'Caa1' 

	#pick out rows where RTG_SP is null but RTG_Moody is not
	I = pd.isnull(d['rtg_sp']) & ~pd.isnull(d['rtg_moody'])
	d.loc[I,'rtg_sp'] = d.loc[I,('rtg_moody')].apply(f)

#	print d.shape[0]
#	#pick only entries with tradetype == Sale_to_Customer
#	I = d['tradetype'] != "Sale_to_Customer"
#	d = d.loc[I]
#	print d.shape[0]

	#drop cusip and name for now
	d = d.drop(['cusip','name','rtg_moody', 'tradetype'], axis = 1)

	#drop rows with missing values for existing data	
	d = d.dropna()
	price = d.pop('price')

	#convert rtg_sp values to numerical values based on standard ranking
	sp_rating_dict = {'AAA':1, 'AA+':2, 'AA':3,
		'AA-':4, 'A+':5, 'A':6, 'A-':7,
		'BBB+':8, 'BBB':9, 'BBB-':10,
		'BB+':11, 'BB':12, 'BB-':13, 'B+':14,
		'B':15, 'B-':16, 'CCC+':17, 'CCC':18,
		'CCC-':19, 'CC':20, 'C':21, 'D':22}
	g = lambda x: sp_rating_dict[x]
	d['rtg_sp'] = d['rtg_sp'].apply(g)
	
	#adjust maturity date data to be days past minimum date in data set
	d['maturity'] = pd.to_datetime(d['maturity'])
	min_maturity_date = d['maturity'].min()
	d['maturity'] = (d['maturity'] - min_maturity_date)/np.timedelta64(1,'D')

	#categorical variables are state, issuetype, issuesource
	le_state = preprocessing.LabelEncoder().fit(d.state) 
	d.state = le_state.transform(d.state)
	
	le_issuetype = preprocessing.LabelEncoder().fit(d.issuetype)
	d.issuetype = le_issuetype.transform(d.issuetype)	

	le_issuesource = preprocessing.LabelEncoder().fit(d.issuesource)
	d.issuesource = le_issuesource.transform(d.issuesource)

	return d, price

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
	d, price = get_data()
	column_names = d.columns

	bins = np.array([ 0, 80, 90, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,112.5,115, 120, 150])
	#bins = np.array([ 0, 80, 90, 95, 98, 100, 102, 105, 110, 120, 150])
	price = np.digitize(price, bins)
	#scale variables
	d = StandardScaler().fit_transform(d)

	#split into test and training parts
	d_train, d_test, p_train, p_test = train_test_split(d, price, test_size=0.20, random_state=13)


#	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#		 "Naive Bayes", "QDA"]

	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
		 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
		 "Naive Bayes", "QDA"]

#	classifiers = [
#	    KNeighborsClassifier(5),
#	    SVC(kernel="linear", C=0.025),
#	    SVC(gamma=2, C=1),
#	    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#	    DecisionTreeClassifier(max_depth=5),
#	    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#	    MLPClassifier(alpha=1),
#	    AdaBoostClassifier(),
#	    GaussianNB(),
#	    QuadraticDiscriminantAnalysis()]

	classifiers = [
	    KNeighborsClassifier(5),
	    SVC(kernel="linear", C=0.025),
	    SVC(gamma=2, C=1),
	    DecisionTreeClassifier(max_depth=None),
	    RandomForestClassifier(max_depth=None, n_estimators=50, max_features=1),
	    MLPClassifier(alpha=1),
	    AdaBoostClassifier(),
	    GaussianNB(),
	    QuadraticDiscriminantAnalysis()]

	for name, clf in zip(names, classifiers):
		clf.fit(d_train, p_train)
		p_pred = clf.predict(d_test)
        	score = clf.score(d_test, p_test)
		print(name + ": number correct predictions/total samples = {}".format(score))
		# Compute confusion matrix
		cnf_matrix = confusion_matrix(p_test, p_pred)
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=bins, title='Confusion matrix, without normalization')
		plt.show()

		

#	gnb = GaussianNB()
#	p_pred = gnb.fit(d_train, p_train).predict(d_test)

#	print("Number of mislabeled prices out of a total {} points : {}".format(d_test.shape[0], (p_test != p_pred).sum()))

#	# Compute confusion matrix
#	cnf_matrix = confusion_matrix(p_test, p_pred)
#	np.set_printoptions(precision=2)

#	# Plot non-normalized confusion matrix
#	plt.figure()
#	plot_confusion_matrix(cnf_matrix, classes=bins,
#		              title='Confusion matrix, without normalization')

#	# Plot normalized confusion matrix
#	plt.figure()
#	plot_confusion_matrix(cnf_matrix, classes=bins, normalize=True,
#		              title='Normalized confusion matrix')

#	plt.show()
	


if __name__ == "__main__":
	main()
