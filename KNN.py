
from collections import Counter
import numpy as np
class KNN():
	def __init__(self, k):
		self.n_components = k
	def __call__(self):
		return self
	def train(self,X,y):
		self.X_train = X
		self.y_train = y
	def distance(self, x):
		dists = np.sqrt(-2 * np.dot(x, self.X_train.T)+ np.sum(self.X_train**2, axis=1) +np.sum(x**2,axis=1)[:,np.newaxis])
		return dists
	#predict the label for passed batch
	def predict(self,x):
		dists = self.distance(x)
		y_pred = np.zeros(dists.shape[0])
		for i in range(dists.shape[0]):
			k_top_y = []
			labels = self.y_train[np.argsort(dists[i, :])].flatten()
			k_top_y = labels[:self.n_components]
			c = Counter(k_top_y)
			y_pred[i] = c.most_common(1)[0][0]
		return(y_pred)
	#compute the accuracy for the prediction
	def accuracy(self,x,y):
		batch_size = 2000
		prediction = []
		for i in range(int(x.shape[0] / (batch_size))):
			y_prime = self.predict(x[i * batch_size:(i+1) * batch_size])
			prediction = prediction + list(y_prime)
		prediction = np.array(prediction, dtype=int)
		y = np.array(y, dtype=int)
		count = np.sum(prediction == y)
		return(float(count)/float(x.shape[0]), prediction)
if __name__ == "__main__":
	KNN()




