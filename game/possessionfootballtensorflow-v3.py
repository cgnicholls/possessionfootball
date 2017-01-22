#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import random
import tensorflow as tf
from collections import deque
import time
import game as g

plt.ion()

# This file implements a simple possession based football drill -- one team
# tries to keep hold of the ball, and is allowed to pass it around, while the
# other team tries to press them and gain possession of the ball.

# First try: the possession team has three players arranged on the vertices of
# an equilateral triangle. If one player is in possession, they choose to pass
# left, right or to hold on to the ball. The ball then travels at some velocity
# in the direction of the player receiving the pass.

# While this is going on, a defender in the middle tries to press the team to
# regain possession. The defender always moves in the direction of the ball, and
# if they get within a certain distance of the ball, they are assumed to have
# won the ball.

# Q-learning parameters
INITIAL_LEARNING_RATE = 1e-2
INITIAL_EPSILON_GREEDY = 1.0
FINAL_EPSILON_GREEDY = 0.0
EPSILON_STEPS = 10000
NUM_ACTIONS = g.NUM_PLAYERS-1
ACTIONS = range(0,g.NUM_PLAYERS-1)
DISCOUNT_FACTOR = 0.7
MINI_BATCH_SIZE = 128
L2_REG = 1e-3
KEEP_PROB = 0.5
THRESHOLD = 10.0
MAX_EPISODE_LENGTH = 2*THRESHOLD
TRAIN_EVERY = 1
NUM_HIDDEN = 30

# Compute the action predicted by the current parameters of the q network for
# the current state.
def compute_action(tf_sess, tf_output_layer,
                   keep_prob, current_state, epsilon_greedy):
    # Choose an action randomly with probability epsilon_greedy.
    if random.random() <= epsilon_greedy:
        action_index = random.randrange(NUM_ACTIONS)
    # Otherwise, choose an action according to the Q-function
    else:
        q_function = tf_sess.run(tf_output_layer, feed_dict={
          tf.get_default_graph().get_tensor_by_name('input/input_layer:0'): [current_state],
          tf.get_default_graph().get_tensor_by_name('keep_prob:0'): keep_prob})[0]
        action_index = np.argmax(q_function)

    # Return the action at action_index
    return ACTIONS[action_index]
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1.0, name='weights')
    weight = tf.Variable(initial)
    return weight, tf.nn.l2_loss(weight)

def bias_variable(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))
    
#def variable_summaries(var, name):
#    with tf.name_scope('summaries'):
#        mean = tf.reduce_mean(var)
#        tf.scalar_summary('mean/' + name, mean)
#        with tf.name_scope('stddev'):
#            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#        tf.scalar_summary('stddev/' + name, stddev)
#        tf.histogram_summary(name, var)
    
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights, l2_weights = weight_variable([input_dim, output_dim])
            #variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            #tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate)
        #tf.histogram_summary(layer_name + '/activations', activations)
        return activations, l2_weights

# We want a network that can well approximate Q-values. In particular, since
# our reward is -1 if we lose possession and otherwise 0, our Q-value will
# always be negative. Thus we don't want to end with a sigmoid!
def create_network(input_dim, num_hidden, num_players):    
    with tf.name_scope('input'):
        input_layer = tf.placeholder(tf.float32, shape=[None, input_dim], name='input_layer')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('layer1'):
        hidden1, l2_hidden1 = nn_layer(input_layer, input_dim, num_hidden, 'layer1')
        #dropped1 = tf.nn.dropout(hidden1, keep_prob)
    with tf.name_scope('layer2'):
        hidden2, l2_hidden2 = nn_layer(hidden1, num_hidden, num_hidden, 'layer2')
        #dropped2 = tf.nn.dropout(hidden2, keep_prob)
    with tf.name_scope('layer3'):
        hidden3, l2_hidden3 = nn_layer(hidden2, num_hidden, num_hidden, 'layer3')
        #dropped3 = tf.nn.dropout(hidden3, keep_prob)
    with tf.name_scope('layer4'):
        hidden4, l2_hidden4 = nn_layer(hidden3, num_hidden, num_players, 'layer4', act=tf.identity)
    with tf.name_scope('output'):
        output_layer = hidden4
    l2_reg = l2_hidden1 + l2_hidden2 + l2_hidden3 + l2_hidden4
    
    return output_layer, l2_reg

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[ACTIONS.index(actions[i])] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

def train(tf_sess, tf_train_operation, tf_cost, l2_reg, tf_output_layer, merged,
        observations, reward_history, keep_prob):
    # Sample a minibatch to train with
    mini_batch = random.sample(observations, MINI_BATCH_SIZE)
    
    states = [d['state'] for d in mini_batch]
    actions = [d['action'] for d in mini_batch]
    rewards = [d['reward'] for d in mini_batch]
    next_states = [d['next_state'] for d in mini_batch]
    expected_q = []

    # Compute Q(s', a'; theta_{i-1}). This is an unbiased estimator for y_i as
    # in DQN paper.
    default_graph = tf.get_default_graph()
    next_q = tf_sess.run(tf_output_layer, 
                         feed_dict={
        default_graph.get_tensor_by_name('input/input_layer:0') : next_states, 
        default_graph.get_tensor_by_name('keep_prob:0'): keep_prob})

    for i in range(len(mini_batch)):
        if mini_batch[i]['terminal']:
            # This was a terminal frame, so the Q-value is just the reward
            expected_q.append(rewards[i])
        else:
            # The Q-value of the current state should be the reward at this
            # timestep plus the maximum of the Q-values at the next state
            expected_q.append(rewards[i] + DISCOUNT_FACTOR * \
                    np.max(next_q[i]))

    one_hot_actions = compute_one_hot_actions(actions)
    
    # Run the train operation to update the q-values towards these q-values
    _, cost, l2_reg_cost, summary = tf_sess.run([tf_train_operation, tf_cost,
        l2_reg, merged], feed_dict={
        default_graph.get_tensor_by_name('input/input_layer:0') : states,
        default_graph.get_tensor_by_name('action:0') : one_hot_actions,
        default_graph.get_tensor_by_name('target:0') : expected_q,
        default_graph.get_tensor_by_name('keep_prob:0') : keep_prob,
        default_graph.get_tensor_by_name('avg_reward/rewards:0'): reward_history})

    return summary, cost, l2_reg_cost

def qlearning():
    tf.reset_default_graph()
    tf_sess = tf.Session()
    
    tf_output_layer, l2_reg = create_network((g.NUM_PLAYERS)*2, NUM_HIDDEN, NUM_ACTIONS)

    tf_action = tf.placeholder("float", [None, NUM_ACTIONS], name='action')

    tf_target = tf.placeholder("float", [None], name='target')

    tf_q_for_action = tf.reduce_sum(tf.mul(tf_output_layer, tf_action),
            reduction_indices=1)

    with tf.name_scope('cost'):
        #reg_losses = [tf.nn.l2_loss(tf.get_variable('layer1/weights'))]
        tf_cost = tf.reduce_mean(tf.square(tf_target - tf_q_for_action)) + \
        l2_reg * L2_REG
        #+ L2_REG * sum(reg_losses)
        tf.scalar_summary('cost', tf_cost)
        tf.scalar_summary('l2_reg', l2_reg)
        #tf.scalar_summary('reg_loss', sum(reg_losses))
        
    with tf.name_scope('avg_reward'):
        tf_rewards = tf.placeholder("float", [None], name='rewards')
        tf_avg_reward = tf.reduce_mean(tf_rewards)
        tf.scalar_summary('avg_reward', tf_avg_reward)

    with tf.name_scope('train_op'):
        tf_train_operation = \
            tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(tf_cost)

    merged = tf.merge_all_summaries()
    # Give this run of the program an identifier
    identifier = str(time.gmtime()[0:5])
    identifier = identifier.replace('(', '').replace(')', '')
    identifier = identifier.replace(' ', '-').replace(',','')

    summarise = False

    if summarise:
        train_writer = tf.train.SummaryWriter('train-' + identifier, tf_sess.graph)
    
    tf_sess.run(tf.initialize_all_variables())

    epsilon_greedy = INITIAL_EPSILON_GREEDY

    transitions = deque()

    episode_lengths = []
    
    ep_index = 0
    loss = None
    
    game = g.Game()
    game.set_render_or_not(False)
    
    current_state = game.reset()
    keep_prob = 0.5
    
    episode_lengths = []
    last_nonzero_rewards = []
    
    t_step = 0
    successful_t_steps = 0
    costs = []
    
    # Record transitions
    while True:
        # Run an episode
        action = compute_action(tf_sess, tf_output_layer,
                                    keep_prob, current_state, epsilon_greedy)
        obs, reward, terminal = game.step_environment(action)
        
        next_state = obs
        last_nonzero_rewards.append(reward)
        last_nonzero_rewards = last_nonzero_rewards[-500:]
        episode_lengths = episode_lengths[-200:]
        
        transitions.append({'state': current_state, 'next_state': next_state, 
        'action': action, 'reward': reward, 'terminal': terminal})

        if terminal or successful_t_steps >= MAX_EPISODE_LENGTH:
            current_state = game.reset()
            episode_lengths.append(successful_t_steps)
            successful_t_steps = 0
        else:
            current_state = next_state
            successful_t_steps += 1

        if len(transitions) > MINI_BATCH_SIZE and t_step % TRAIN_EVERY == 0:
            summary, cost, l2_reg_cost = train(tf_sess, tf_train_operation, tf_cost,
                    l2_reg, tf_output_layer, merged, transitions,
                    last_nonzero_rewards[-500:], KEEP_PROB)
            costs.append(cost)
            costs = costs[-500:]
            if t_step % 100 == 0:
                if summarise:
                    train_writer.add_summary(summary, t_step)
        t_step = t_step + 1
        epsilon_greedy = epsilon_greedy - \
        (INITIAL_EPSILON_GREEDY-FINAL_EPSILON_GREEDY) / float(EPSILON_STEPS)
        epsilon_greedy = max(FINAL_EPSILON_GREEDY, epsilon_greedy)

        avg_nonzero_reward = np.mean(last_nonzero_rewards)
        if (ep_index % 100) == 0:
            print "Average nonzero reward", avg_nonzero_reward, "Std:", np.std(last_nonzero_rewards), "Average ep length:", np.mean(episode_lengths)
            if len(costs) > 0:
                print "Average cost", np.mean(costs), "l2_reg", l2_reg_cost * L2_REG
            print "Playing randomly with prob", epsilon_greedy
        ep_index = ep_index + 1
        
        if ep_index > 1000:
            if np.mean(episode_lengths) > THRESHOLD:
                print "Min reward over last 500 is", avg_nonzero_reward, ">", THRESHOLD, ", so finished training"
                return tf_sess, tf_output_layer

tf_sess, tf_output_layer = qlearning()

# Now observe the game with the learned parameters
game = g.Game()
game.set_render_or_not(True)
last_nonzero_rewards = []
current_state = game.reset()
print "Obs", current_state
successful_t_steps = 0
for i in range(1000):
    action = compute_action(tf_sess, tf_output_layer, 1.0,
                            current_state, 0.0)
    #action = np.random.randint(NUM_ACTIONS)
    obs, reward, terminal = game.step_environment(action)
    current_state = obs
    last_nonzero_rewards.append(reward)
    last_nonzero_rewards = last_nonzero_rewards[-500:]
    print obs, reward, terminal
    print "Average rewards:", np.mean(last_nonzero_rewards)
    
    if terminal or successful_t_steps > 10:
        print "Resetting"
        current_state = game.reset()
        successful_t_steps = 0
    else:
        successful_t_steps += 1
