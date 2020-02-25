from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time

# DATASET
path_to_file = './shakespeare.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text
dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# MODEL
vocab_size = len(vocab) #65
embedding_dim = 256
rnn_units = 1024
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
	model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim,
							batch_input_shape=[batch_size, None]),
	tf.keras.layers.LSTM(rnn_units,
						return_sequences=True,
						stateful=True,
						recurrent_initializer='glorot_uniform'),
	tf.keras.layers.Dense(vocab_size)
	])
	return model
model = build_model(
	vocab_size = vocab_size,
	embedding_dim=embedding_dim,
	rnn_units=rnn_units,
	batch_size=BATCH_SIZE)

# TRAIN
EPOCHS=10
checkpoint_dir = './high_level_LSTM'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

def train(id):
	if (id==1):
		train1()
	elif (id==2):
		train2()
	else:
		raise Exception('L\'id deve essere 1 o 2')

def train1():
	def loss(labels, logits):
		return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
	model.compile(optimizer='adam', loss=loss)
	checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_prefix,
		save_weights_only=True)

	history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

def train2():
	optimizer = tf.keras.optimizers.Adam()
	#optimizer = tf.keras.optimizers.SGD(0.1)
	def train_step(inp, target):
		with tf.GradientTape() as tape:
			predictions = model(inp)
			loss = tf.reduce_mean(
				tf.keras.losses.sparse_categorical_crossentropy(
					target, predictions, from_logits=True))
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		return loss

	for epoch in range(EPOCHS):
		start = time.time()
		# initializing the hidden state at the start of every epoch
		# initally hidden is None
		hidden = model.reset_states()
		loss = -1

		for (batch_n, (inp, target)) in enumerate(dataset):
			loss = train_step(inp, target)

			if batch_n % 100 == 0:
				template = 'Epoch {} Batch {} Loss {}'
				print(template.format(epoch+1, batch_n, loss))

		# saving (checkpoint) the model every 5 epochs
		if (epoch + 1) % 5 == 0:
			model.save_weights(checkpoint_prefix.format(epoch=epoch))

		print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
		print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

	model.save_weights(checkpoint_prefix.format(epoch=epoch))

train(2)

# INFERE
def generate_text(model, start_string):
	# Number of characters to generate
	num_generate = 1000
	input_eval = [char2idx[s] for s in start_string]
	input_eval = tf.expand_dims(input_eval, 0)
	text_generated = []
	# Low temperatures results in more predictable text.
	# Higher temperatures results in more surprising text.
	# Experiment to find the best setting.
	temperature = 1.0

	model.reset_states()
	for i in range(num_generate):
		predictions = model(input_eval)
		predictions = tf.squeeze(predictions, 0)
		predictions = predictions / temperature
		predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
		input_eval = tf.expand_dims([predicted_id], 0)
		text_generated.append(idx2char[predicted_id])
	return (start_string + ''.join(text_generated))

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
print("Generating text...")
print(generate_text(model, start_string=u"WARWICK:"))
#print(generate_text(model, start_string=u"First Citizen:"))
print("============================================================================")
print("============================================================================")
print("============================================================================")
print(generate_text(model, start_string=u"F"))
