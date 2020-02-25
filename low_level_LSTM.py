import tensorflow as tf
import numpy as np
import os

### DATASET
data = open('shakespeare.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
T_steps = 100
training_examples_number = data_size//(T_steps+1)
pointer_MAX = training_examples_number*(T_steps+1)
print("data has %d characters, %d unique" % (data_size, vocab_size))
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}
data = [char_to_idx[x] for x in data]
dataset = np.zeros((vocab_size,pointer_MAX),dtype=np.float64)
for index in range(pointer_MAX):
	dataset[data[index], index] = 1



### MODEL
X_size = vocab_size
H_size = 100 # Size of the hidden layer
learning_rate = 1e-1 # Learning rate
weight_sd = 0.1 # Standard deviation of weights for initialization
z_size = H_size + X_size # Size of concatenate(H, X) vector

class Model(object):
	def __init__(self):
		# Initialize the weights
		self.W_f = tf.Variable(np.random.randn(H_size, z_size) * weight_sd + 0.5)
		self.b_f = tf.Variable(np.zeros((H_size, 1)))

		self.W_i = tf.Variable(np.random.randn(H_size, z_size) * weight_sd + 0.5)
		self.b_i = tf.Variable(np.zeros((H_size, 1)))

		self.W_C = tf.Variable(np.random.randn(H_size, z_size) * weight_sd)
		self.b_C = tf.Variable(np.zeros((H_size, 1)))

		self.W_o = tf.Variable(np.random.randn(H_size, z_size) * weight_sd + 0.5)
		self.b_o = tf.Variable(np.zeros((H_size, 1)))

		#For final layer to predict the next character
		self.W_v = tf.Variable(np.random.randn(X_size, H_size) * weight_sd)
		self.b_v = tf.Variable(np.zeros((X_size, 1)))

		self.reset_states()
		self.trainable_variables = [self.W_f, self.b_f, self.W_i, self.b_i, self.W_C, self.b_C, self.W_o, self.b_o, self.W_v, self.b_v]

	def reset_states(self):
		self.h_prev = np.zeros((H_size, 1))
		self.C_prev = np.zeros((H_size, 1))

	def __call__(self, inputs):
		assert inputs.shape == (vocab_size, T_steps)
		outputs = []
		for time_step in range(T_steps):
			x = inputs[:,time_step]
			x = tf.expand_dims(x,axis=1)

			z = tf.concat([self.h_prev,x],axis=0)

			f = tf.matmul(self.W_f, z) + self.b_f
			f = tf.sigmoid(f)

			i = tf.matmul(self.W_i, z) + self.b_i
			i = tf.sigmoid(i)

			o = tf.matmul(self.W_o, z) + self.b_o
			o = tf.sigmoid(o)

			C_bar = tf.matmul(self.W_C, z) + self.b_C
			C_bar = tf.tanh(C_bar)

			C = (f * self.C_prev) + (i * C_bar)

			h = o * tf.tanh(C)

			v = tf.matmul(self.W_v, h) + self.b_v
			v = tf.sigmoid(v)

			y = tf.math.softmax(v, axis=0)

			self.h_prev = h
			self.C_prev = C

			outputs.append(y)
		outputs = tf.squeeze(tf.stack(outputs,axis=1))
		return outputs

	def generate(self, input_):
		self.reset_states()
		num_generate = 1000
		outputs = []
		x = input_
		for time_step in range(num_generate):
			z = tf.concat([self.h_prev,x],axis=0)

			f = tf.matmul(self.W_f, z) + self.b_f
			f = tf.sigmoid(f)

			i = tf.matmul(self.W_i, z) + self.b_i
			i = tf.sigmoid(i)

			o = tf.matmul(self.W_o, z) + self.b_o
			o = tf.sigmoid(o)

			C_bar = tf.matmul(self.W_C, z) + self.b_C
			C_bar = tf.tanh(C_bar)

			C = (f * self.C_prev) + (i * C_bar)

			h = o * tf.tanh(C)

			v = tf.matmul(self.W_v, h) + self.b_v
			v = tf.sigmoid(v)

			y = tf.math.softmax(v, axis=0)

			self.h_prev = h
			self.C_prev = C

			index = np.random.choice(range(vocab_size), p=y.numpy().ravel())

			x = np.zeros((vocab_size, 1))
			x[index] = 1

			outputs.append(idx_to_char[index])
		outputs = ''.join(c for c in outputs)
		return outputs


model = Model()

def compute_loss(predictions, desired_outputs):
	l = 0
	for i in range(T_steps):
		l -= tf.math.log(predictions[desired_outputs[i], i])
	return l


### TRAIN
EPOCHS=10
checkpoint_dir = './low_level_LSTM'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

optimizer = tf.keras.optimizers.Adam()
#optimizer = tf.keras.optimizers.SGD(0.1)

print("STARTING TRAINING.......................")

def train_step(inputs, desired_outputs):
	with tf.GradientTape() as tape:
		predictions = model(inputs)
		loss = compute_loss(predictions, desired_outputs)
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss


for epoch in range(EPOCHS):
	model.reset_states()
	epoch_loss = 0
	print ('Epoch',epoch+1,'of',EPOCHS)
	for pointer in range(0, pointer_MAX, T_steps+1):
		inputs = dataset[:, pointer: pointer + T_steps]
		desired_outputs = data[pointer + 1: pointer + T_steps + 1]
		epoch_loss += train_step(inputs,desired_outputs)
		#print ('Pointer {} Loss {:.4f}'.format(pointer, epoch_loss))
	print ('Epoch {} Loss {:.4f}'.format(epoch+1, epoch_loss))

print("done.")

input_ = dataset[:, 0]
input_ = tf.expand_dims(input_,axis=1)
print(model.generate(input_))
