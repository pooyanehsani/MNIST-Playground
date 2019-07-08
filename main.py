from KNN import KNN
from read_data import read_MNIST
from NCA import NCA
from sklearn import metrics
from SVM import SVM
from PCA import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import timeit

def choose_k(x,y,xt,yt,min,max):
	index = 0
	acc_max = 0
	for i in range(min,max):
		knn = KNN(i)
		knn.train(x, y)
		acc = knn.accuracy(xt, yt)
		if acc > acc_max:
			index = i
			acc_max = acc
	return index, acc_max

def twod_plot(x, y):
	unique = list(set(y))
	colors = [plt.cm.jet(float(i) / max(unique)) for i in unique]
	for i, u in enumerate(unique):
		xi = [x[j,0] for j in range(len(x)) if y[j] == u]
		x2i = [x[j,1] for j in range(len(x)) if y[j] == u]
		plt.scatter(xi, x2i, c=colors[i], label=str(u))
	plt.legend()
	plt.tight_layout()
	plt.show()

def threeD_plot(x,y):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
	plt.show()

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0,1,2,3,4,5,6,7,8,9]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


(x_train, y_train), (x_test, y_test) = read_MNIST()

#result for the row data
index , _ = choose_k(x_train,y_train,x_test,y_test,5,8)
print index
knn = KNN(index)
knn.train(x_train,y_train)
acc , pred = knn.accuracy(x_test,y_test)
print("KNN acc on the original data",acc)
plot_confusion_matrix(y_test, pred,
                      title='KNN Confusion matrix original data')
plt.show()
print("knn report", metrics.classification_report(y_test,pred))
s = SVM()
s.train(x_train,y_train,x_test,y_test)
acc, _, pred = s.predict(x_test,y_test)
print("svm acc on original data", acc)
plot_confusion_matrix(y_test, pred,
                      title='svm Confusion matrix')
plt.show()
print("svm report", metrics.classification_report(y_test,pred))
#results for 50 pca
start = timeit.default_timer()
pca = PCA(50)
(x_train_trans, x_test_trans) = pca.fit_transform(x_train,x_test)
stop = timeit.default_timer()
print('Time: ', stop - start)
#KNN
index , _ = choose_k(x_train_trans, y_train, x_test_trans, y_test, 5,8)
print index
knn = KNN(index)
knn.train(x_train_trans,y_train)
acc , pred = knn.accuracy(x_test_trans,y_test)
print("KNN acc on the PCA data",acc)
plot_confusion_matrix(y_test, pred,
                     title='Confusion matrix PCA data')
plt.show()
print("pca + knn report", metrics.classification_report(y_test,pred))
#SVM
s = SVM()
s.train(x_train_trans,y_train,x_test_trans,y_test)
acc, _, pred = s.predict(x_test_trans,y_test)
print("svm acc on pca data", acc)
plot_confusion_matrix(y_test, pred,
                      title='pca + svm Confusion matrix')
plt.show()
print("pca + svm report", metrics.classification_report(y_test,pred))

#results for 50 NCA
nca = NCA(50)
nca.fit(x_train[:10000],y_train[:10000])
x_train_trans = nca.transform(x_train)
x_test_trans = nca.transform(x_test)
index , _ = choose_k(x_train_trans, y_train, x_test_trans, y_test, 5,10)
print index
knn = KNN(index)
knn.train(x_train_trans,y_train)
acc , pred = knn.accuracy(x_test_trans,y_test)
print("KNN acc on the NCA data",acc)
plot_confusion_matrix(y_test, pred,
                      title='NCA + kNN Confusion matrix')
plt.show()
print("NCA + knn report", metrics.classification_report(y_test,pred))
s = SVM()
s.train(x_train_trans,y_train,x_test_trans,y_test)
acc, _, pred = s.predict(x_test_trans,y_test)
print("svm acc on pca data", acc)
plot_confusion_matrix(y_test, pred,
                      title='NCA + svm Confusion matrix')
plt.show()
print("NCA + svm report", metrics.classification_report(y_test,pred))

