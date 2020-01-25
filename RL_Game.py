# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:31:51 2020

@author: Emin
"""

"""
RL Game
"""
# pygame template
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pygame
import random

# window size

width  = 360
height = 360
fps    = 30 # how fast game is

white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(blue)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, red, self.rect.center, self.radius)
        self.rect.centerx = width / 2
        self.rect.bottom = height - 1
        self.speedx = 0
        
    def update(self, action):
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -6
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = +6
        else:
            self.speedx = 0
            
        self.rect.x += self.speedx
        
        if self.rect.right > width:
            self.rect.right = width
        if self.rect.left < 0:
            self.rect.left = 0
            
    def getCordinates(self):
        return (self.rect.x, self.rect.y)
    
    
class Enemy(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(red)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image, blue, self.rect.center, self.radius)
        
        self.rect.x = random.randrange(0, width - self.rect.width)
        self.rect.y = random.randrange(2, 6)
        
        self.speedx = 0
        self.speedy = 10
        
    def update(self):
        #self.rect.x += self.speedx
        self.rect.y += self.speedy
        
        if self.rect.top > height + 10:
            self.rect.x = random.randrange(0, width - self.rect.width)
            self.rect.y = random.randrange(2, 6)
            self.speedy = 10
 

    def getCordinates(self):
        return (self.rect.x, self.rect.y)   
        
        
class DQLAgent:
    def __init__(self):
        
        #paremeter/hyperparameter
        
        self.state_size  = 4 # distance
        self.action_size = 3 # right, left, no move
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 0.3
        self.epsilon_decay = 0.7
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
    
    def build_model(self):
        # neural network for deep a learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(self.action_size, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action,reward,next_state,done):
        #storage
        self.memory.append((state, action,reward,next_state,done))
    
    def act(self,state):
        #acting : explore or exploit
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        #training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action,reward,next_state,done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state, train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        


class Env(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.all_sprite.add(self.player)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        
        self.reward = 0
        self.reward_total = 0
        self.done = False
        self.agent = DQLAgent()
        
    def FindDistance(self, a, b):
        d = a - b
        return d
        
        
    def step(self, action):
        state_list = []
        
        #update
        self.player.update(action)
        self.enemy.update()
        
        # get coordinate
        next_player_state = self.player.getCordinates()
        next_m1_state = self.m1.getCordinates()
        next_m2_state = self.m2.getCordinates()
        
        # find distance
        
        state_list.append(self.FindDistance(next_player_state[0], next_m1_state[0]))
        state_list.append(self.FindDistance(next_player_state[1], next_m1_state[1]))
        state_list.append(self.FindDistance(next_player_state[0], next_m2_state[0]))
        state_list.append(self.FindDistance(next_player_state[1], next_m2_state[1]))
        
        return [state_list]
        
        
        
    # reset 
    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.all_sprite.add(self.player)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        
        self.reward = 0
        self.reward_total = 0
        self.done = False
        
        state_list = []
        
        # get coordinate
        player_state = self.player.getCordinates()
        m1_state = self.m1.getCordinates()
        m2_state = self.m2.getCordinates()
        
        # find distance
        
        state_list.append(self.FindDistance(player_state[0], m1_state[0]))
        state_list.append(self.FindDistance(player_state[1], m1_state[1]))
        state_list.append(self.FindDistance(player_state[0], m2_state[0]))
        state_list.append(self.FindDistance(player_state[1], m2_state[1]))
        
        return [state_list]
        
        
    def run(self):
        # game loop
        state = self.initialStates()
        running = True
        batch_size = 24
        while running:
            self.reward = 2
            # keep loop running at the right speed
            clock.tick(fps)
            #process input
    
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.reward_total += self.reward
                    
            hits = pygame.sprite.spritecollide(self.player, self.enemy, False, pygame.sprite.collide_circle)
            if hits:
                self.reward = -150
                self.reward_total += self.reward
                self.done = True
                running = False
                print("Total reward", self.reward_total)
            # storage    
            self.agent.remember(state, action, self.reward, next_state, self.done)
            
            # update state
            
            state = next_state
            
            # training
            self.agent.replay(batch_size)
            # epsilon greed
            self.agent.adaptiveEGreedy()
            # draw / render (show)
            screen.fill(green)
            self.all_sprite.draw(screen)
    
            # after drawing flip display
            pygame.display.flip()
            
            
        pygame.quit()
            
            

        
if __name__ == "__main__":
    
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1
        print("Episode : ", t)
        liste.append(env.reward_total)
            
        # initialize pygame and create window
        
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Sercan Saglam = Cesur ")
        clock = pygame.time.Clock()
        
        env.run()

# sprite

#all_sprite = pygame.sprite.Group()
#enemy = pygame.sprite.Group()
#player = Player()
#m1 = Enemy()
#m2 = Enemy()
#all_sprite.add(player)
#enemy.add(m1)
#enemy.add(m2)
#all_sprite.add(m1)
#all_sprite.add(m2)




# game loop

#running = True
#while running:
#    # keep loop running at the right speed
#    clock.tick(fps)
#    
#    #process input
#    
#    for event in pygame.event.get():
#        if event.type == pygame.QUIT:
#            running = False
#            
##    # update
##    
##    all_sprite.update()
#    hits = pygame.sprite.spritecollide(player, enemy, False, pygame.sprite.collide_circle)
#    if hits:
#        running = False
#        print("Game_over")
    # draw / render(show)
#    screen.fill(green)
#    all_sprite.draw(screen)
#    
#    # after drawing flip display
#    pygame.display.flip()
#            
#            
#pygame.quit()           
