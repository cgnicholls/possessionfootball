import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
import random

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
RENDER = True
VERBOSE = False
PROB_PASS = 0.9
NUM_PLAYERS = 5
TIME_STEP = 0.1
PLAYER_RADIUS = 0.1
CIRCLE_RADIUS = 2.0
POSSESSION_DISTANCE = 2*PLAYER_RADIUS
BALL_SPEED = 0.1
DEFENDER_SPEED = 0.05

# Arrange the players on the unit circle
def init_player_positions(num_players, circle_radius):
    p_positions = np.zeros((num_players, 2))
    for player in range(num_players):
        theta = (float(player) / float(num_players)) * (2*np.pi)
        p_positions[player, :] = circle_radius * np.array([np.cos(theta),
            np.sin(theta)])
    return p_positions

ball_position = np.array([0,0])
player_positions = init_player_positions(NUM_PLAYERS, CIRCLE_RADIUS)
defender_position = np.mean(player_positions)

ball_position = player_positions[0]
ball_velocity = np.array([0,0])

possession = True
player_in_possession = None
last_pass = None

FIG = plt.figure()
AXES = plt.axes(xlim = (-3,3), ylim = (-3, 3))

def render_environment(p_positions, d_position, b_position):
    plt.cla()
    defender = plt.Circle(defender_position, radius = POSSESSION_DISTANCE/2, fc='g')
    ball = plt.Circle(ball_position, radius = POSSESSION_DISTANCE/2, fc='b')
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
def pass_ball(p1, p2):
    direction = player_positions[p2] - player_positions[p1]
    length_2 = squared_length(direction)
    assert squared_distance > 0
    return direction / np.sqrt(length_2)

t = 0
while possession:
    
    # The ball is in some position. If it is within defender_distance of the
    # defender, then the defender wins
    if squared_distance(ball_position, defender_position) < POSSESSION_DISTANCE**2:
        possession = False
        break
    
    # Otherwise, check if one of the players is in possesion
    for player in range(NUM_PLAYERS):
        if (squared_distance(player_positions[player], ball_position) <
                POSSESSION_DISTANCE**2) and (last_pass != player):
            print "In possession of player", player
            player_in_possession = player
            break
    
    # If a player is in possession, then the ball stops moving and has the
    # position of the player. The player in possession can also choose to pass
    # the ball, or hold onto it. This is done randomly at the moment.
    if player_in_possession != None:
        print "PLAYER IN POSSESSION"
        ball_position = player_positions[player_in_possession]
        
        # If we pass the ball, then update the ball's velocity
        if random.random() < PROB_PASS:
            print "PASSING THE BALL"
            players_to_receive = [(player_in_possession + i) % NUM_PLAYERS for i
                    in range(1, NUM_PLAYERS)]
            player_to_receive = random.choice(players_to_receive)
            print "player to receive", player_to_receive
            ball_velocity = pass_ball(player_in_possession, player_to_receive)
            last_pass = player_in_possession
            player_in_possession = None

    # Now update the environment
    defender_position, ball_position = step_environment(TIME_STEP,
            defender_position, ball_position, ball_velocity)

    if t % RENDER_EVERY == 0:
        if RENDER:
            render_environment(player_positions, defender_position, ball_position)
    if VERBOSE:
        print "Positions"
        print defender_position
        print ball_position
        print ball_velocity
        print player_in_possession
        print last_pass

    t = t + 1

        
print "Defender gained possession!"

