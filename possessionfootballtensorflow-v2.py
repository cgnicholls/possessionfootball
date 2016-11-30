#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:06:48 2016

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import random
import tensorflow as tf
from collections import deque
import time

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

RENDER_EVERY = 50
RENDER = False
VERBOSE = False
PROB_PASS = 0.9
NUM_PLAYERS = 5
TIME_STEP = 1.0
PLAYER_RADIUS = 0.1
CIRCLE_RADIUS = 3
POSSESSION_DISTANCE = 2*PLAYER_RADIUS
BALL_SPEED = 0.11
DEFENDER_SPEED = 0.1

# Q-learning parameters
INITIAL_LEARNING_RATE = 1e-3
INITIAL_EPSILON_GREEDY = 1.0
FINAL_EPSILON_GREEDY = 0.0
EPSILON_STEPS = 1000
NUM_ACTIONS = NUM_PLAYERS-1
ACTIONS = range(0,NUM_PLAYERS-1)
DISCOUNT_FACTOR = 0.7
MINI_BATCH_SIZE = 128
L2_REG = 1e-3
KEEP_PROB = 0.5

np.random.seed(1)
# Arrange the players randomly
INITIAL_PLAYER_POSITIONS = np.random.rand(NUM_PLAYERS, 2) * CIRCLE_RADIUS - CIRCLE_RADIUS / 2

class Game():
    def __init__(self):
        self.player_positions = None
        self.ball_position = None
        self.defender_position = None
        self.state = None
        self.ball_velocity = None
        self.last_player = None
        self.playing_game = False
        self.t = None
        self.fig = None
        self.axes = None
        
    def rl_state(self, player, reward):
        state = np.zeros((2+NUM_PLAYERS-1,2))
        state[0,:] = self.ball_position
        state[1,:] = self.defender_position
        indices = range(0, player) + range(player+1,NUM_PLAYERS)
        state[2:,:] = self.player_positions[indices]
        #state[2:,:] = self.player_positions
        state = state.ravel()
        
        terminal = not self.playing_game
        return state, reward, terminal
        
    def init_game(self):
        self.player_positions = init_players_on_circle(NUM_PLAYERS, CIRCLE_RADIUS)
        self.defender_position = np.mean(self.player_positions, 0) + \
        CIRCLE_RADIUS / 5.0 * np.random.randn(2)
        starting_player = np.random.randint(NUM_PLAYERS)
        self.ball_position = self.player_positions[starting_player]
        self.ball_velocity = np.array([0,0])
        self.playing_game = True
        self.last_player = None
        self.action_just_taken = False
        self.t = 0
        return self.rl_state(starting_player,0)[0]
        
    def reset(self):
        return self.init_game()
        
    def set_render_or_not(self, render_or_not):
        self.render_or_not = render_or_not
        if self.fig == None and self.render_or_not == True:
            self.fig = plt.figure()
            self.axes = plt.axes(xlim = (-CIRCLE_RADIUS-1,CIRCLE_RADIUS+1), ylim = (-CIRCLE_RADIUS-1, CIRCLE_RADIUS+1))
        
    def render(self):
        plt.cla()
        defender = plt.Circle(self.defender_position, radius = POSSESSION_DISTANCE/2, fc='g')
        ball = plt.Circle(self.ball_position, radius = POSSESSION_DISTANCE/2, fc='b')
        for player in range(NUM_PLAYERS):
            player_circle = plt.Circle(self.player_positions[player], radius =
                                       POSSESSION_DISTANCE/2, fc='r')
            plt.gca().add_patch(player_circle)
        plt.gca().add_patch(defender)
        plt.gca().add_patch(ball)
        plt.pause(0.001)
        plt.draw()
        
    def stop_game(self):
        self.playing_game = False
        
    def step_environment(self, action):
        # If the game state is not started, then start the game
        if self.playing_game == False:
            print "Game not playing. Reset environment"
            return
        # Enter game loop
        action_used = False
        while True:
            # Resolve interceptions
            # If the defender has intercepted the ball, then the game is over, and we
            # return the rl state
            if squared_distance(self.defender_position, self.ball_position) < POSSESSION_DISTANCE**2:
                self.stop_game()
                return self.rl_state(0, -1)
             
            # Make sure the ball is still in the circle
            assert squared_length(self.ball_position) < CIRCLE_RADIUS**2 * 2
            
            action_taken = False
            # Resolve if a player is in possession. If so, the agent has to choose
            # an action!
            for player in range(NUM_PLAYERS):
                if squared_distance(self.player_positions[player], self.ball_position) < \
                POSSESSION_DISTANCE**2 and self.action_just_taken == False:
                    # If we haven't used the action yet, then apply it, else
                    # request a new action.
                    if action_used == False:
                        self.pass_to_player(player, action)
                        action_used = True
                        self.last_player = player
                        self.action_just_taken = True
                        action_taken = True
                    else:
                        return self.rl_state(player, 0)
                    break
            if action_taken == False:
                self.action_just_taken = False

            # Update the environment
            self.update_environment(TIME_STEP)
            
            # Render if required
            if self.render_or_not and self.t % RENDER_EVERY == 0:
                self.render()

    def update_environment(self, time_step):
        # Update defender position towards the ball
        self.defender_position = self.defender_position + \
        time_step * DEFENDER_SPEED * (self.ball_position - self.defender_position) / \
        np.sqrt(squared_distance(self.ball_position, self.defender_position))
        self.ball_position = self.ball_position + time_step * BALL_SPEED * self.ball_velocity

    def pass_to_player(self, player, action):
        if action >= player:
            player_receiving = action + 1
        else:
            player_receiving = action
        direction = self.player_positions[player_receiving] - self.ball_position
        length = np.sqrt(squared_length(direction))
        if length > POSSESSION_DISTANCE:
            self.ball_velocity = direction / length
        else:
            self.ball_velocity = np.zeros(2)
            
def init_players_on_circle(num_players, circle_radius):
    p_positions = np.zeros((num_players, 2))
    for player in range(num_players):
        theta = (float(player) / float(num_players)) * (2*np.pi)
        p_positions[player, :] = circle_radius * np.array([np.cos(theta),
        np.sin(theta)])
    return p_positions 

# Arrange the players randomly on the unit circle
def init_player_positions(num_players, circle_radius):
#    p_positions = np.zeros((num_players, 2))
#    for player in range(num_players):
#        #theta = (float(player) / float(num_players)) * (2*np.pi)
#        #p_positions[player, :] = circle_radius * np.array([np.cos(theta),
#        #    np.sin(theta)])
#        p_positions[player, :] = np.random.rand(2) * circle_radius - circle_radius/2 * np.array([1,1])
#    return p_positions
    return INITIAL_PLAYER_POSITIONS

def squared_distance(v1, v2):
    return np.sum((v1-v2)**2)

def squared_length(v):
    return np.sum(v**2)

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
    initial = tf.truncated_normal(shape, stddev=0.1, name='weights')
    weight = tf.Variable(initial)
    return weight, tf.nn.l2_loss(weight)

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
    
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.histogram_summary(name, var)
    
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
        dropped1 = tf.nn.dropout(hidden1, keep_prob)
    with tf.name_scope('layer2'):
        hidden2, l2_hidden2 = nn_layer(dropped1, num_hidden, num_hidden, 'layer2')
        dropped2 = tf.nn.dropout(hidden2, keep_prob)
    with tf.name_scope('layer3'):
        hidden3, l2_hidden3 = nn_layer(dropped2, num_hidden, num_players, 'layer3', act=tf.identity)
    with tf.name_scope('output'):
        output_layer = hidden3
    l2_reg = l2_hidden1 + l2_hidden2 + l2_hidden3
    return output_layer, l2_reg

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[ACTIONS.index(actions[i])] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

def train(tf_sess, tf_train_operation, tf_output_layer, merged, observations,
          reward_history, keep_prob):
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
            expected_q.append(rewards[i] + DISCOUNT_FACTOR * \
                    np.max(next_q[i]))

    one_hot_actions = compute_one_hot_actions(actions)
    
    # Run the train operation to update the q-values towards these q-values
    _, summary = tf_sess.run([tf_train_operation, merged], feed_dict={
        default_graph.get_tensor_by_name('input/input_layer:0') : states,
        default_graph.get_tensor_by_name('action:0') : one_hot_actions,
        default_graph.get_tensor_by_name('target:0') : expected_q,
        default_graph.get_tensor_by_name('keep_prob:0') : keep_prob,
        default_graph.get_tensor_by_name('avg_reward/rewards:0'): reward_history})

    return summary

def qlearning():
    tf.reset_default_graph()
    tf_sess = tf.Session()
    
    tf_output_layer, l2_reg = create_network((1+NUM_PLAYERS)*2, 20, NUM_ACTIONS)

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
    train_writer = tf.train.SummaryWriter('train-' + identifier, tf_sess.graph)
    
    tf_sess.run(tf.initialize_all_variables())

    epsilon_greedy = INITIAL_EPSILON_GREEDY

    transitions = deque()

    episode_lengths = []
    
    ep_index = 0
    loss = None
    
    game = Game()
    game.set_render_or_not(False)
    
    current_state = game.reset()
    keep_prob = 0.5
    
    last_nonzero_rewards = []
    
    t_step = 0
    
    # Record transitions
    while True:
        # Run an episode
        action = compute_action(tf_sess, tf_output_layer,
                                    keep_prob, current_state, epsilon_greedy)
        obs, reward, terminal = game.step_environment(action)
        #print "Observation", obs
        
        next_state = obs
        last_nonzero_rewards.append(reward)
        last_nonzero_rewards = last_nonzero_rewards[-500:]
        
        transitions.append({'state': current_state, 'next_state': next_state, 
        'action': action, 'reward': reward, 'terminal': terminal})

        if terminal:
            current_state = game.reset()
        else:
            current_state = next_state
        
        if len(transitions) > MINI_BATCH_SIZE:
            summary = train(tf_sess, tf_train_operation, 
                         tf_output_layer, merged, transitions, 
                         last_nonzero_rewards[-500:], KEEP_PROB)
            if t_step % 100 == 0:
                train_writer.add_summary(summary, t_step)
            t_step = t_step + 1
        epsilon_greedy = epsilon_greedy - \
        (INITIAL_EPSILON_GREEDY-FINAL_EPSILON_GREEDY) / float(EPSILON_STEPS)
        epsilon_greedy = max(FINAL_EPSILON_GREEDY, epsilon_greedy)

        avg_nonzero_reward = np.mean(last_nonzero_rewards)
        if (ep_index % 100) == 0:
            print "Average nonzero reward", avg_nonzero_reward
            print "Playing randomly with prob", epsilon_greedy
        ep_index = ep_index + 1
        
        if ep_index > 1000:
            if avg_nonzero_reward > -0.1:
                print "Min reward over last 500 is", avg_nonzero_reward, "> -0.1, so finished training"
                return tf_sess, tf_output_layer

tf_sess, tf_output_layer = qlearning()

# Now observe the game with the learned parameters
game = Game()
game.set_render_or_not(True)
last_nonzero_rewards = []
current_state = game.reset()
print "Obs", current_state
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
    
    if terminal:
        #print "Resetting"
        current_state = game.reset()
        #print "Obs", current_state
