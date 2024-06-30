from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
INF = 1e12

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        try:
            self.q_table = np.load(file_name, allow_pickle=True)
        except Exception as e:
            self.q_table = np.empty((83000, 2), dtype=object)
            self.q_table[:] = ""
            # TODO: Initialize Q-table

        self.lr = 0.1 # TODO: Learning rate
        self.discount_factor = 0.9 # TODO: Discount factor
        self.epsilon = 0.05 # TODO: Epsilon
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.min_lr = 0.1
        self.lr_decay = 0.995
        self.last_action = UP

        self.num_wall_loss = 0
        self.num_enemy_loss = 0 
        self.num_self_loss = 0


    def add_to_q_table(self, state):
        empty_indices = np.where(self.q_table[:, 0] == "")[0]
        first_empty = empty_indices[0]
        self.q_table[first_empty, 0] = state
        self.q_table[first_empty, 1] = [0]*4
        return first_empty

    def opposite(self, side1, side2):
        return (
            (side1 == UP and side2 == DOWN) or
            (side1 == DOWN and side2 == UP) or
            (side1 == RIGHT and side2 == LEFT) or
            (side1 == LEFT and side2 == RIGHT))

    def get_optimal_policy(self, state):
        indices = np.where(self.q_table[:, 0] == state)[0]
        index = indices[0]
        action = np.argmax(self.q_table[index, 1])
        if (self.opposite(action, self.last_action)):
            copy = self.q_table[index, 1][:]
            copy[action] = -INF
            action = np.argmax(copy)
        return action

    def make_action(self, state):
        indices = np.where(self.q_table[:, 0] == state)[0]
        if indices.size <= 0:
            self.add_to_q_table(state)
        
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
            while self.opposite(action, self.last_action):
                action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        return action

    def update_q_table(self, state, action, next_state, reward):
        ns_indices = np.where(self.q_table[:, 0] == next_state)[0]
        ns_index = -1
        if ns_indices.size <= 0:
            ns_index = self.add_to_q_table(next_state)
        else:
            ns_index = ns_indices[0]
        
        s_indices = np.where(self.q_table[:, 0] == state)[0]
        s_index = s_indices[0]

        best_next_action = np.argmax(self.q_table[ns_index, 1])
        td_target = reward + self.discount_factor * self.q_table[ns_index, 1][best_next_action]
        td_error = td_target - self.q_table[s_index, 1][action]
        self.q_table[s_index, 1][action] += self.lr * td_error

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        if self.lr > self.min_lr:
            self.lr *= self.lr_decay

    def calc_enemy_direction(self, enemy):
        head_x = self.head.pos[0]
        head_y = self.head.pos[1]
        min_y_distance = 21
        min_x_distance = 21
        x_direction = -1
        y_direction = -1
        too_close = False

        for cube in enemy.body:
            cube_x = cube.pos[0]
            distance = abs(cube_x - head_x)
            if distance < min_x_distance:
                min_x_distance = distance
                if cube_x < head_x:
                    x_direction = LEFT
                else:
                    x_direction = RIGHT

        for cube in enemy.body:
            cube_y = cube.pos[1]
            distance = abs(cube_y - head_y)
            if distance < min_y_distance:
                min_y_distance = distance
                if cube_y < head_y:
                    y_direction = UP
                else:
                    y_direction = DOWN
        
        if min_x_distance < 3 and min_y_distance < 3:
            too_close = True
        return (x_direction, y_direction, too_close)

    def calc_snack_direction(self, snack):
        head_x = self.head.pos[0]
        head_y = self.head.pos[1]
        snack_x = snack.pos[0]
        snack_y = snack.pos[1]
        x_distance = abs(head_x - snack_x)
        y_distance = abs(head_y - snack_y)
        snack_direction = -1
        too_close = False

        if x_distance >= y_distance:
            if snack_x > head_x:
                snack_direction = RIGHT
            else:
                snack_direction = LEFT
        else:
            if snack_y > head_y:
                snack_direction = DOWN
            else:
                snack_direction = UP

        if x_distance < 3 and y_distance < 3:
            too_close = True
        
        return (snack_direction, too_close)

    def calc_wall_direction(self):
        head_x = self.head.pos[0]
        head_y = self.head.pos[1]
        closest_wall = -1
        too_close = False
        
        if head_x <= head_y and head_x <= 20-head_y:
            if head_x < 3: too_close = True
            closest_wall = LEFT
        if head_x >= head_y and head_x <= 20-head_y:
            if head_y < 3: too_close = True
            closest_wall = UP
        if head_x <= head_y and head_x >= 20-head_y:
            if 20-head_y < 3: too_close = True
            closest_wall = DOWN
        else:
            if 20-head_x < 3: too_close = True
            closest_wall = RIGHT
        return (closest_wall, too_close)

    def on_the_border(self, pos):
        x = pos[0]
        y = pos[1]
        if x >= ROWS - 1 or x < 1 or y >= ROWS - 1 or y < 1:
            return True
        return False
    
    def get_adjacent_space(self, enemy, snack):
        head_x = self.head.pos[0]
        head_y = self.head.pos[1]
        adj_states = [0]*4
        x_deltas = [0, 1, 0, -1]
        y_deltas = [-1, 0, 1, 0]

        for i in range(4):
            x = head_x + x_deltas[i]
            y = head_y + y_deltas[i]
            if snack.pos[0] == x and snack.pos[1] == y:
                adj_states[i] = 1
            for cube in enemy.body:
                if cube.pos[0] == x and cube.pos[1] == y:
                    adj_states[i] = 2
            for cube in self.body:
                if cube.pos[0] == x and cube.pos[1] == y:
                    adj_states[i] = 2
            if self.on_the_border((x, y)):
                adj_states[i] = 2
        return adj_states

    def create_state(self, snack, other_snake):
        other_snake_direction = self.calc_enemy_direction(other_snake)
        snack_direction = self.calc_snack_direction(snack)
        closest_wall_direction = self.calc_wall_direction()
        adjacents = self.get_adjacent_space(other_snake, snack)
        state = [
            other_snake_direction[0], 
            other_snake_direction[1], 
            int(other_snake_direction[2]),
            snack_direction[0],
            int(snack_direction[1]), 
            closest_wall_direction[0], 
            int(closest_wall_direction[1]), 
            self.last_action,
            ]
        state += adjacents
        for i in range(len(state)):
            state[i] = str(state[i])
        return ' '.join(state)

    def move(self, snack, other_snake):
        state = self.create_state(snack, other_snake) # TODO: Create state
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.last_action = LEFT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.last_action = RIGHT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.last_action = UP
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.last_action = DOWN
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        next_state = self.create_state(snack, other_snake)
        return state, next_state, action
        # TODO: Create new state after moving and other needed values and return them
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False

    def move_toward_snack(self, snack):
        snack_direction = self.calc_snack_direction(snack)
        last_action = self.last_action
        if last_action == snack_direction[0]:
            if snack_direction[1]:
                return 100
            return 60
        return 0

    def move_away_from_snack(self, snack):
        snack_direction = self.calc_snack_direction(snack)
        if self.opposite(self.last_action, snack_direction[0]):
            if snack_direction[1]:
                return 100
            return 60
        return 0
    
    def move_toward_wall(self):
        wall_direction = self.calc_wall_direction()
        closest_wall = wall_direction[0]
        too_close = wall_direction[1]
        return (too_close and closest_wall == self.last_action)
    
    def move_away_from_wall(self):
        wall_direction = self.calc_wall_direction()
        closest_wall = wall_direction[0]
        too_close = wall_direction[1]
        return (too_close and self.opposite(closest_wall, self.last_action))
    
    def move_toward_other_snake(self, other_snake):
        other_snake_direction = self.calc_enemy_direction(other_snake)
        enemy_x = other_snake_direction[0]
        enemy_y = other_snake_direction[1]
        too_close = other_snake_direction[2]
        return (too_close and (enemy_x == self.last_action or enemy_y == self.last_action))
    
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False
        
        if self.check_out_of_board():
            # TODO: Punish the snake for getting out of the board
            self.num_wall_loss += 1
            reward -= 2500
            win_other = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 200
            # TODO: Reward the snake for eating
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # TODO: Punish the snake for hitting itself
            self.num_self_loss += 1
            reward -= 2000
            win_other = True
            reset(self, other_snake)

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            
            if self.head.pos != other_snake.head.pos:
                # TODO: Punish the snake for hitting the other snake
                self.num_enemy_loss += 1
                reward -= 1500
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    # TODO: Reward the snake for hitting the head of the other snake and being longer
                    reward += 700
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    # TODO: No winner
                    pass
                else:
                    # TODO: Punish the snake for hitting the head of the other snake and being shorter
                    reward -= 700
                    win_other = True
                    
            reset(self, other_snake)

        toward_reward = self.move_toward_snack(snack)
        reward += toward_reward

        away_reward = self.move_away_from_snack(snack)
        reward -= away_reward
        
        if self.move_toward_wall():
            reward -= 300
        
        if self.move_away_from_wall():
            reward += 50
        
        if self.move_toward_other_snake(other_snake):
            reward -= 300

        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        