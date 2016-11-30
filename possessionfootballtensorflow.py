import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import random
import tensorflow as tf
from collections import deque

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

RENDER_EVERY = 20
RENDER = False
VERBOSE = False
PROB_PASS = 0.9
NUM_PLAYERS = 5
TIME_STEP = 0.1
PLAYER_RADIUS = 0.1
CIRCLE_RADIUS = 1.5
POSSESSION_DISTANCE = 2*PLAYER_RADIUS
BALL_SPEED = 0.1
DEFENDER_SPEED = 0.1

# Q-learning parameters
INITIAL_LEARNING_RATE = 1e-4
INITIAL_EPSILON_GREEDY = 1.0
FINAL_EPSILON_GREEDY = 0.01
EPSILON_STEPS = 1000
NUM_ACTIONS = NUM_PLAYERS-1
ACTIONS = range(1,NUM_PLAYERS)
DISCOUNT_FACTOR = 0.7
MINI_BATCH_SIZE = 512
L2_REG = 1e-3
KEEP_PROB = 0.5

# Arrange the players on the unit circle
def init_player_positions(num_players, circle_radius):
    p_positions = np.zeros((num_players, 2))
    for player in range(num_players):
        theta = (float(player) / float(num_players)) * (2*np.pi)
        p_positions[player, :] = circle_radius * np.array([np.cos(theta),
            np.sin(theta)])
    return p_positions

if RENDER:
    FIG = plt.figure()
    AXES = plt.axes(xlim = (-3,3), ylim = (-3, 3))

def render_environment(p_positions, d_position, b_position):
    plt.cla()
    defender = plt.Circle(d_position, radius = POSSESSION_DISTANCE/2, fc='g')
    ball = plt.Circle(b_position, radius = POSSESSION_DISTANCE/2, fc='b')
    for player in range(NUM_PLAYERS):
        player_circle = plt.Circle(p_positions[player], radius =
                POSSESSION_DISTANCE/2, fc='r')
        plt.gca().add_patch(player_circle)

    plt.gca().add_patch(defender)
    plt.gca().add_patch(ball)
    plt.pause(0.001)
    plt.draw()

def squared_distance(v1, v2):
    return np.sum((v1-v2)**2)

def squared_length(v):
    return np.sum(v**2)

def step_environment(timestep, d_position, b_position, b_velocity):
    d_position = d_position + timestep * DEFENDER_SPEED * (b_position -
            d_position) / np.sqrt(squared_distance(b_position, d_position))
    b_position = b_position + timestep * BALL_SPEED * b_velocity
    return d_position, b_position

# Returns the ball velocity resulting from p1 passing to p2
def pass_ball(p_positions, p1, p2):
    assert p1 != p2
    direction = p_positions[p2] - p_positions[p1]
    length_2 = squared_length(direction)
    assert squared_distance > 0
    return direction / np.sqrt(length_2)

# Transforms so that the ith player (i = p_possession) is at the coordinate of
# p_positions[0]. Also renumber the actions so that it is as if the 0th player
# is playing instead of the ith player.
def get_state(p_positions, d_position, p_possession):
    num_players = len(p_positions)
    try:
        theta = -(float(p_possession) / float(num_players)) * (2*np.pi)
    except TypeError:
        print "TypeError: p_possession", p_possession, "num_players", num_players
        theta = 0.0
    # Rotate d_position by theta about the origin
    return np.dot(d_position, np.array([[np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]]))

# Compute the action predicted by the current parameters of the q network for
# the current state.
def compute_action(tf_sess, tf_input_layer, tf_output_layer, tf_keep_prob,
                   keep_prob, current_state, epsilon_greedy):
    # Choose an action randomly with probability epsilon_greedy.
    if random.random() <= epsilon_greedy:
        action_index = random.randrange(NUM_ACTIONS)
    # Otherwise, choose an action according to the Q-function
    else:
        q_function = tf_sess.run(tf_output_layer, feed_dict={
          tf_input_layer: [current_state],
          tf_keep_prob: keep_prob})[0]
        action_index = np.argmax(q_function)

    # Return the action at action_index
    return ACTIONS[action_index]

# We now want to sample q-values from the game. Need to collect [s, a, r, s',
# terminal] tuples. We return these now, as well as the score. We represent the
# state as the position of the defender, the action as the number of the player
# that we pass to, and the reward is +1 if we complete the pass, and -1 if it is
# intercepted. To give each player the same network, we transform the
# coordinates so that the player is at a fixed position.
def play_game(tf_sess, tf_input_layer, tf_output_layer, tf_keep_prob, keep_prob,
              epsilon_greedy, render=RENDER, maxlen=500):
    t = 0
    ball_position = np.array([0,0])
    player_positions = init_player_positions(NUM_PLAYERS, CIRCLE_RADIUS)
    defender_position = np.mean(player_positions, 0) + \
    CIRCLE_RADIUS / 3.0 * np.random.randn(2)

    ball_position = player_positions[0]
    ball_velocity = np.array([0,0])
    player_in_possession = None
    last_pass = None

    transitions = []
    current_state = None

    while len(transitions) < maxlen:

        # Otherwise, check if one of the players is in possesion
        for player in range(NUM_PLAYERS):
            if (squared_distance(player_positions[player], ball_position) <
                    POSSESSION_DISTANCE**2) and (last_pass != player):
                # If the current state is none, then we have just started the
                # game, and should update the current state to be the current
                # defender's position.
                if current_state is None:
                    current_state = get_state(player_positions,
                            defender_position, player)

                # Otherwise, someone previously passed the ball, and a new
                # player received it. Thus we should add a successful
                # transition. player_in_possession is the person who passed the
                # ball, and we still need the defender's position with respect
                # to him.
                else:
                    next_state = get_state(player_positions, defender_position,
                            player)
                    transitions.append({'state': current_state, 'action':
                        action, 'reward': 0, 'next_state': next_state,
                        'terminal': False})
                    if VERBOSE:
                        print "Received pass:", player
                player_in_possession = player
                break
        
        # If a player is in possession, then the ball stops moving and has the
        # position of the player. The player in possession can also choose to pass
        # the ball, or hold onto it. This is done randomly at the moment.
        if player_in_possession != None:
            ball_position = player_positions[player_in_possession]

            # Now make a decision using the neural network
            current_state = get_state(player_positions, defender_position, player_in_possession)

            action = compute_action(tf_sess, tf_input_layer, tf_output_layer,
                                    tf_keep_prob, keep_prob, current_state, epsilon_greedy)
            player_to_receive = (player_in_possession + action) % NUM_PLAYERS
            if VERBOSE:
                print "Player to receive:", player_to_receive
                print "Action:", action
#            # If we pass the ball, then update the ball's velocity
#            if random.random() < PROB_PASS:
#                players_to_receive = [(player_in_possession + i) % NUM_PLAYERS for i
#                        in range(1, NUM_PLAYERS)]
#                player_to_receive = random.choice(players_to_receive)

            if player_to_receive != player_in_possession:
                ball_velocity = pass_ball(player_positions, player_in_possession,
                        player_to_receive)
                last_pass = player_in_possession
                player_in_possession = None
            else:
                # Else we just held on to the ball, so there should be some time penalty for this
                last_pass = player_in_possession
                
        # The ball is in some position. If it is within defender_distance of the
        # defender, then the defender wins and we return the number of passes
        if squared_distance(ball_position, defender_position) < POSSESSION_DISTANCE**2:
            if last_pass is None:
                print "Last pass shouldn't be None here!"
                # As a hack, we just put last_pass = player_in_possession
                last_pass = player_in_possession
            next_state = get_state(player_positions, defender_position,
                    last_pass)
            transitions.append({'state': current_state, 'action': action,
                'reward': -1, 'next_state': next_state, 'terminal': True})
            return transitions

        # Now update the environment
        defender_position, ball_position = step_environment(TIME_STEP,
                defender_position, ball_position, ball_velocity)

        if t % RENDER_EVERY == 0:
            if render:
                render_environment(player_positions, defender_position, ball_position)
        if VERBOSE:
            print "Positions"
            print defender_position
            print ball_position
            print ball_velocity
            print player_in_possession
            print last_pass

        t = t + 1
    return transitions

def create_network(num_hidden, num_players):
    input_layer = tf.placeholder(tf.float32, shape=[None,2])
    keep_prob = tf.placeholder(tf.float32)
    W1 = tf.Variable(tf.truncated_normal(shape=[2, num_hidden],
        stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    hidden1 = tf.nn.relu(tf.matmul(input_layer, W1) + b1)
    hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
    
    W2 = tf.Variable(tf.truncated_normal(shape=[num_hidden, num_hidden],
        stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
    hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    
    W3 = tf.Variable(tf.truncated_normal(shape=[num_hidden, num_players],
        stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[num_players]))
    output_layer = tf.nn.softmax(tf.matmul(hidden2_drop, W3) + b3)
    
    l2_norm = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    
    return input_layer, output_layer, keep_prob, l2_norm

# Return a one hot vector with a 1 at the index for the action.
def compute_one_hot_actions(actions):
    one_hot_actions = []
    for i in xrange(len(actions)):
        one_hot = np.zeros([NUM_ACTIONS])
        one_hot[ACTIONS.index(actions[i])] = 1
        one_hot_actions.append(one_hot)
    return one_hot_actions

def train(tf_sess, observations, tf_input_layer, tf_output_layer,
        tf_train_operation, tf_action, tf_target, tf_cost, tf_keep_prob,
        keep_prob):
    # Sample a minibatch to train with
    mini_batch = random.sample(observations, MINI_BATCH_SIZE)
    
    states = [d['state'] for d in mini_batch]
    actions = [d['action'] for d in mini_batch]
    rewards = [d['reward'] for d in mini_batch]
    next_states = [d['next_state'] for d in mini_batch]
    expected_q = []

    # Compute Q(s', a'; theta_{i-1}). This is an unbiased estimator for y_i as
    # in DQN paper.
    next_q = tf_sess.run(tf_output_layer, feed_dict={tf_input_layer :
        next_states, tf_keep_prob: keep_prob})

    for i in range(len(mini_batch)):
        if mini_batch[i]['terminal']:
            # This was a terminal frame, so the Q-value is just the reward
            expected_q.append(rewards[i])
        else:
            expected_q.append(rewards[i] + DISCOUNT_FACTOR * \
                    np.max(next_q[i]))

    one_hot_actions = compute_one_hot_actions(actions)
    
    # Run the train operation to update the q-values towards these q-values
    _, loss = tf_sess.run([tf_train_operation, tf_cost], feed_dict={
        tf_input_layer : states,
        tf_action : one_hot_actions,
        tf_target : expected_q,
        tf_keep_prob : keep_prob})

    return loss

def qlearning():
    tf_sess = tf.Session()
    
    tf_input_layer, tf_output_layer, tf_keep_prob, tf_l2_norm = create_network(50, NUM_ACTIONS)

    tf_action = tf.placeholder("float", [None, NUM_ACTIONS])

    tf_target = tf.placeholder("float", [None])

    tf_q_for_action = tf.reduce_sum(tf.mul(tf_output_layer, tf_action),
            reduction_indices=1)

    tf_cost = tf.reduce_mean(tf.square(tf_target - tf_q_for_action)) + \
    L2_REG * tf_l2_norm

    tf_train_operation = \
            tf.train.AdamOptimizer(INITIAL_LEARNING_RATE).minimize(tf_cost)

    tf_sess.run(tf.initialize_all_variables())

    epsilon_greedy = INITIAL_EPSILON_GREEDY

    transitions = deque()

    episode_lengths = []
    
    ep_index = 0
    loss = None
    
    # Record transitions
    while True:
        # Run an episode
        maxlen = 10
        episode_data = play_game(tf_sess, tf_input_layer, tf_output_layer,
                tf_keep_prob, KEEP_PROB, epsilon_greedy, maxlen=maxlen)
        #print "Length of episode", len(episode_data)
        for ep in episode_data:
            transitions.append(ep)
        episode_lengths.append(len(episode_data))
        if len(transitions) > MINI_BATCH_SIZE:
            loss = train(tf_sess, transitions, tf_input_layer, tf_output_layer,
                    tf_train_operation, tf_action, tf_target, tf_cost, tf_keep_prob,
                    KEEP_PROB)
        epsilon_greedy = epsilon_greedy - \
        (INITIAL_EPSILON_GREEDY-FINAL_EPSILON_GREEDY) / float(EPSILON_STEPS)
        epsilon_greedy = max(FINAL_EPSILON_GREEDY, epsilon_greedy)
        #print epsilon_greedy

        avg_ep_length = np.mean(episode_lengths[-50:])
        if (ep_index % 20) == 0:
            print "Average number of passes for last 50 games:", avg_ep_length
            print "Q-learning loss", loss
            print "Playing randomly with prob", epsilon_greedy
        ep_index = ep_index + 1
        
        if ep_index > 0:
            if avg_ep_length > 8:
                print "Min reward over last 50 is", avg_ep_length, "> 9, so finished training"
                return tf_sess, tf_input_layer, tf_output_layer, tf_keep_prob

tf_sess, tf_input_layer, tf_output_layer, tf_keep_prob = qlearning()

# Now observe the game with the learned parameters
FIG = plt.figure()
AXES = plt.axes(xlim = (-3,3), ylim = (-3, 3))
for i in range(100):
    play_game(tf_sess, tf_input_layer, tf_output_layer, tf_keep_prob, 1.0, 0.0, render=True, maxlen=10)
