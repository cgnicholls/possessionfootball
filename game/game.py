import numpy as np

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
        state = np.zeros((1+NUM_PLAYERS-1,2))
        state[0,:] = self.defender_position
        indices = range(0, player) + range(player+1,NUM_PLAYERS)
        state[1:,:] = self.player_positions[indices]
        
        # We subtract the ball position from the positions of all players 
        # and the defender. This reduces the size of the state
        state = state - self.ball_position
        state = state.ravel()
        
        terminal = not self.playing_game
        return state, reward, terminal
        
    def init_game(self):
        self.player_positions = init_player_positions(NUM_PLAYERS, CIRCLE_RADIUS)
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
