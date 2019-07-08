import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
class SVM():
	#initialize the batch size and learning rate for the training
	def __init__(self, batch_size = 1000, learning_rate = 0.001):
		self.lr = learning_rate
		self.bs = batch_size

	def __call__(self):
		return self
	#train procedure
	#
	def train(self,X,y,X_val,y_val):
		self.X = X
		self.y = y#to_categorical(y)
		self.W = np.random.rand(self.X.shape[1], 10)
		loss_plot = np.empty([0,3])
		for i in range(int((self.X.shape[0]) / (self.bs))):
			X_batch = self.X[i * self.bs:(i+1) * self.bs]
			y_batch = self.y[i * self.bs:(i+1) * self.bs]
			loss, grad = self.svm_loss_vectorized(self.W,X_batch,y_batch)
			self.W += -self.lr * grad
			print("batch number %s" %i)
			val_accu , val_loss, pre = self.predict(X_val,y_val)
			loss_plot = np.append(loss_plot, np.array([[i,loss,val_loss]]), axis=0)
	def svm_loss_vectorized(self, W, X, y, reg = 1e2, delta=0.05):

		loss = 0.0
		dW = np.zeros(W.shape)  # initialize the gradient as zero
		num_train = X.shape[0]

		scores = np.dot(X, W)

		correct_class_scores = np.choose(y, scores.T).reshape(-1, 1)


		margins = np.maximum(scores - correct_class_scores + delta, 0.0)

		margins[np.arange(num_train), y] = 0.0

		loss = np.sum(margins) / float(num_train)
		loss += 0.5 * reg * np.sum(W * W)
		grad_mask = (margins > 0).astype(int)
		grad_mask[np.arange(y.shape[0]), y] = - np.sum(grad_mask, axis=1)
		dW = np.dot(X.T, grad_mask)

		dW /= float(num_train)
		dW += reg * W
		return loss, dW
	def predict(self, X, y, delta=0.01):
		scores = X.dot(self.W)
		correct_class_scores = np.choose(y, scores.T).reshape(-1, 1)
		margins = np.maximum(scores - correct_class_scores + delta, 0.0)
		margins[np.arange(X.shape[0]), y] = 0.0
		loss = np.sum(margins) / float(X.shape[0])
		pred = scores.argmax(axis = 1)
		count = np.sum(pred == y)
		return(float(count)/float(X.shape[0])*100, loss, pred)
if __name__ == "__main__":
	SVM()
