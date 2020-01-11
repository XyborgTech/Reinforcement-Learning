import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() #this is cuz of a tensorflow error, so it lets me use tf.placeholder
import gym
import numpy as np

'''
REINFORCE-Monte-Carlo Policy Gradient
'''

state = tf.placeholder(shape=[None, 4], dtype=tf.float32) #placeholder 4 element vector state
rewards=tf.placeholder(shape=[None], dtype=tf.float32)
actions=tf.placeholder(shape=[None], dtype=tf.int32)

W=tf.Variable(tf.random_uniform([4,64], dtype=tf.float32)) #random variables
#tf.Variable() with random values in array of 4 x 64 and same type as rest of variables
hidden=tf.nn.relu(tf.matmul(state, W))
#tf.nn.relu is the Rectified Linear Unit activation function, defined as (f(x) = max(0,x), with inputs (features, name=None)
#and outputs postive matrix values. tf.matmul multiplies matrix a * matrix b for matrix c
O=tf.Variable(tf.random_uniform([64,2], dtype=tf.float32))
output=tf.nn.softmax(tf.matmul(hidden,O))
#multiplaction of hidden (which is relu(matmul(state,W)) and O which is random.
#softmax is then calculated probabilites of selection of action/output
#softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
indices=tf.range(0, tf.shape(output)[0]) * 2 + actions
#this returns the outputs * 2 + actions, and since actions currently empty, fills them; stored information
actProbs= tf.gather(tf.reshape(output, [-1]), indices)
#tf.reshape(tensor, new_shape); tf.gather(params-tensor with values, indices-must be in range [params.shape[axis])
#it outputs a tensor same type as params; so combines output and indices(actions) and outputs their probabilities?
loss = -tf.reduce_mean(tf.log(actProbs)*rewards)
#this is the loss function for future rewards, computes the mean of the inputed tensor
trainOp = tf.train.AdamOptimizer(.0001).minimize(loss)
#optimizes training and initiales variables

gamma = 0.98 #discount factor gamma
env = gym.make('CartPole-v0') #env. for PoC do MountainCar
sess = tf.Session() #allows execution of graohs and holds values
sess.run(tf.global_variables_initializer()) #variables initialized now randomly after AdamOptimizer added
history = []
score = 0

'''
sess.run() > initilized first above, then either assigns new value or returns current value below
'''

total=5001 #10001
batch_size = 50 #50
steps = 1000 #1000

'''so for 1000 iterations, resets env, then for 100 iterations runs policy gradient algorithm, then in batches of 50
prints average score and resets it for new cycle'''
for i in range(total):
    s = env.reset()
    '''Policy Gradient Algorithm, Monte Carlo cuz computes after end of episode, run in batches to optimize and 
    decrease variance; first gets probability of action and randomly picks an action. then gathers data from the
    environment and assigns next_s=observation, r=reward, dn=done, _, based on the random action. 
    ***adds next_s, act, r to a history, which is important so that it can keep track of what action led to what 
    reward, this is stored in an array. tbe  adds reward to the score, makes s = observation'''
    for j in range(steps):
        act_prob = sess.run(output, feed_dict={state: [s]})
        act = np.random.choice(range(2), p=act_prob[0])
        '''np.random.choice(array/int, size=None, replace=True, p=None) where p is probabilities associated 
        with each entry in the array. if not given, assumes uniform distribution. thus the act is assigned 
        a random choice of either 0 or 1, with the probabilities with variables initilized from state and output
        >remember output was calculated as softmax'''
        next_s, r, dn, _ = env.step(act) #observes 1 step
        history.append((s, act, r)) #history size varies, since episodes may terminate earlier. this would lead to
        #dn=True and AdamOptimizer called, and then new episode beginning till steps complete
        score += r
        s = next_s
        '''here, once done is True, new R variable, then it looks at the most recent state, action, reward, and 
        stores to corresponding arrays. then runs session with AdamOptimizer and the stored states, actions, rewards
        History is then cleared, because the AdamOptimizer has the data, and the loop breaks to repeat
        j loop again. same thing again, but this new data is fed into AdamOptimizer as well'''
        if dn:
            R = 0
            for s, act, r in history[::-1]: #history[::-1] means reverse the array
                R = r + gamma * R #discount factor applied
                feed_dict = {state: [s], actions: [act], rewards: [R]}
                sess.run(trainOp, feed_dict)
                '''now a lot happens within the above line, and it is the meat of the ascend gradient policy
                trainOp> trains data using AdamOptimizer, and minimizes loss using -reduce_mean(log_function)
                and be run with the given states'''
            history=[]
            break
    #after certain # of episodes, print score and reset score
    if i%batch_size == 0:
        print("episode {} avg steps : {}".format(i, score/batch_size))
        score = 0

'''
https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/AdamOptimizer
AdamOptimizer Explained- is an extension to stochastic gradient descent
parameters: (lr/step_size-is proportion the steps are updated, exponential decay rates for first moment estimates,
exponential decay rate for second moment estimates (should be set close to 1 depending on problem), epsilon-very small 
number to prevent division by zero, decay-learning rate decay)
default parameters in tensorflow: (0.001, 0.9, 0.999, 1e-08)
tf.train.AdamOptimizer().minimize()- minimize() calls compute_gradients(returns a list of gradient pairs)
then apply_gradients(returns an Operation that applies gradient specificed (for us loss, the negative log))
'''