'''
#Implement REINFORCE
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #this is cuz of a tensorflow error, so it lets me use tf.placeholder
import gym
import numpy as np

state = tf.placeholder(shape=[None, 4], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
rewards = tf.placeholder(shape=[None], dtype=tf.float32)

lr = 0.0001

#Policy Gradient Learning
W = tf.Variable(tf.random_uniform([4,64], dtype=tf.float32)) #randomly filled array 4x64
hidden = tf.nn.relu(tf.matmul(state, W)) #multiplies matrices and makes all values positive, negative ones to 0
O = tf.Variable(tf.random_uniform([64,2], dtype=tf.float32)) #randomly filled array 64x2
output = tf.nn.softmax(tf.matmul(hidden, O)) #softmax calculates probabilites of selection of action/output
indices=tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs= tf.gather(tf.reshape(output, [-1]), indices) #action probabilites
loss = -tf.reduce_mean(tf.log(actProbs)*rewards) #loss function for future rewards, computes mean of log tensor
trainOp = tf.train.AdamOptimizer(lr).minimize(loss) #AdamOptimizer function with minimized loss using gradient descent

gamma = 0.98
env = gym.make('CartPole-v0')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

history = []
score = 0

games = 10001
game_batch = 50
game_steps = 1000

for i in range(games):
	s = env.reset()

	for j in range(game_steps):
		act_prob = sess.run(output, feed_dict={state: [s]})
		act = np.random.choice(range(2), p=act_prob[0])

		next_s, r, dn, _ = env.step(act)
		history.append((s, act, r))

		score += r
		s = next_s

		if dn:
			R = 0
			for s, act, r in history[::-1]:
				R = r + gamma * R
				feed_dict = {state: [s], actions: [act], rewards:[R]}

				sess.run(trainOp, feed_dict)

			history=[]
			break

	if i%game_batch == 0:
		print('Episode', i, 'with average steps', score/game_batch)
		score = 0
