"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
The cart pole example. Policy is oscillated.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
#OUTPUT_GRAPH = False
MAX_EPISODE = 5001
#DISPLAY_REWARD_THRESHOLD = 1000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 500   # maximum time step in one episode
#RENDER = False  # rendering wastes time

GAMMA = 0.95     # reward discount in TD error
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001     # learning rate for critic

env = gym.make('CartPole-v1')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error

episodes = []
reward = []

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

ave_score = 0

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        #if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            ave_score = ave_score + running_reward
            
            print("episode:", i_episode, "  reward:", int(running_reward))
            if i_episode%50==0:
                episodes.append(i_episode)
                reward.append(ave_score)
                ave_score = 0
            break



np.savetxt("CartPoleData_ActorCritic1.txt", np.transpose([episodes, reward]), fmt="%.3f")

""" 
 #PsuedoCode Actor-Critic#
 
'''
 
 initialize state and policy
 start with a sample a according to a multiplied return policy
 
 for each step in 1 episode
	sample reward and transition to new state
	sample action based on returned state and action
	
	CRITIC compute value based on reward + discounted state/action - previous state/action estimate
	ACTOR update policy using policy gradient method (AdamOptimizer prob)
	
	CRITIC update weights with linear TD(0) equation
	
	assign a < a' and s < s'
	
	loop
'''

import gym
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(2)
tf.set_random_seed(2)

MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

env.seed(1)  # reproducible
env = env.unwrapped

n_actions = env.action_space.n

s = tf.placeholder(shape=[None, 4], dtype=tf.float32)
a = tf.placeholder(tf.int32, None, "act")

episodes = 3000

lr = 0.01
LR_A = 0.001
LR_C = 0.01


###Actor Policy Gradient###
td_error = tf.placeholder(tf.float32, None, "td_error")

l1 = tf.layers.dense(
				inputs=s,
				units=20,    # number of hidden units
				activation=tf.nn.relu,
				kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
				bias_initializer=tf.constant_initializer(0.1),  # biases
				name='l1'
			)

acts_prob = tf.layers.dense(
				inputs=l1,
				units=n_actions,    # output units
				activation=tf.nn.softmax,   # get action probabilities
				kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
				bias_initializer=tf.constant_initializer(0.1),  # biases
				name='acts_prob'
			)
			
log_prob = tf.log(acts_prob[0, a])
exp_v = tf.reduce_mean(log_prob * td_error)
train_op = tf.train.AdamOptimizer(lr).minimize(-exp_v)


for i in range(episodes):
	state = env.reset()
	t=0
	track_r = []
	
	while True:
		act_prob = sess.run(acts_prob, feed_dict={state: [s]})
		a = np.random.choice(range(2), p=act_prob[0])
		
		s_, r, done, _ = env.step(a)
		
		if done: r = -20
		
		track_r.append(r)
		
		#critic.learn
		s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
		v_ = sess.run(v, {s: s_})
		td_error, _ = sess.run([td_error, train_op],
										  {s: s, v_: v_, r: r})

		#actor.learn
		s = s[np.newaxis, :]
		feed_dict = {s: s, a: a, td_error: td}
		_, exp_v = sess.run([train_op, exp_v], feed_dict)

		s = s_
		t += 1
		
		if done or t >= 1000:
			ep_rs_sum = sum(track_r)
			
			if 'running_reward' not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
			
			print("episode:", i_episode, " reward:", int(running_reward))
			break
		




"""