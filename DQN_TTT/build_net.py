from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import rmsprop
from keras.optimizers import RMSprop, SGD
import numpy as np
from experience_replay import ER
from DGame import TicTacToes
import random
import matplotlib.pyplot as plt
import copy
import pygame
from Game import TicTacToe, Humanplayer, Randomplayer
from QLearning import  Qlearning

class Bot:
    
    def __init__(self, max_steps = 5):
        self.update_weight = 100
        self.memory_max_len = 500
        self.epsilon = 1.0
        self.ER = ER(self.memory_max_len)
        self.max_steps = max_steps
        self.ttt = TicTacToes()
        self.player = 'O'
        self.bot = 'X'
        self.learning_rate = 0.001
        self.gamma = 0.97

        self.game = TicTacToe() #game instance
        self.player1=Humanplayer() #human player
        self.player2=Qlearning()  #agent
        self.game.startGame(self.player1,self.player2)#player1 is X, player2 is 0

        self.train_model = Sequential()
        ''''
        self.train_model.add(Conv2D(input_shape = (None, 3, 3, 3), filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
        self.train_model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
        self.train_model.add(Flatten())
        self.train_model.add(Dense(units=243, activation = 'relu'))
        self.train_model.add(Dense(units=9, activation = 'softmax'))
        '''
        self.train_model.add(Dense(units=243,input_dim=27, activation = 'relu'))
        self.train_model.add(Dense(units=243, activation = 'relu'))
        self.train_model.add(Dense(units=9, activation = 'linear'))
        self.train_model.compile(loss='mse', optimizer = RMSprop(lr = self.learning_rate), metrics=['accuracy'])
        #self.train_model.compile(loss='mse', optimizer = SGD(lr = self.learning_rate), metrics=['accuracy'])

        
        self.frozen_model = Sequential()
        '''
        self.frozen_model.add(Conv2D(input_shape = (None, 3, 3, 3), filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
        self.frozen_model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
        self.frozen_model.add(Flatten())
        self.frozen_model.add(Dense(units=243, activation = 'relu'))
        self.frozen_model.add(Dense(units=9, activation = 'softmax'))
        '''
        self.frozen_model.add(Dense(units=243,input_dim=27, activation = 'relu'))
        self.frozen_model.add(Dense(units=243, activation = 'relu'))
        self.frozen_model.add(Dense(units=9, activation = 'linear'))
        self.frozen_model.compile(loss='mse', optimizer = RMSprop(lr = self.learning_rate))
        
    def clip(self, x): #clip reward to rannge [1, -1]
        if (x > 1):
            return 1
        elif x < -1:
            return -1
        else:
            return x
    
    def frozen_update(self):
        weights = self.train_model.get_weights()
        target_weights = self.frozen_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.frozen_model.set_weights(target_weights)

    def get_highest_score_valid_move(self, states, actions):
        s = []
        for i in range(len(actions)):
            s.append(states[i+18])
        for move_choice in actions.argsort()[::-1]:
            if s[move_choice] == 1:
                return move_choice
    def possible_moves(self):
        return [moves + 1 for moves, v in enumerate(self.ttt.get_board()) if v == ' ']

    def train(self, max_episodes):
        greedy = 0
        winner = 0
        lost = 0
        draw = 0
        total_step = 0
        win_result = []
        lost_result = []
        draw_result = []
        total_rate = 0
        for episode in range(max_episodes):
            state, legal_moves = self.ttt.reset()
            print("trainining", episode)
            if(episode % self.update_weight == 0):
                print('save weight')
                self.frozen_update()
                self.train_model.save_weights('weights.h5', overwrite=True)
                #self.frozen_model.compile(loss='mse', optimizer ='rmsprop')
            #random variable
            a = random.random()
            '''
            if(episode >=0):
                _action = self.ttt.get_random_move()                  
                self.ttt.drawMove(_action, self.player)
            '''
            for step in range(self.max_steps):
            #while(True):
                state = copy.deepcopy(state)
                pre_reversed_state = copy.deepcopy(self.ttt.get_bot_move())
                inp = np.array([state])
                preds = self.train_model.predict(inp)[0]
                if a < self.epsilon:
                    action = self.ttt.get_random_move()
                else:
                    action = self.get_highest_score_valid_move(state, preds)
                    greedy += 1
                    
                    #action = np.argmax(preds)
                
                reward, done = self.ttt.drawMove(action, self.bot)
                self.epsilon = max(self.epsilon * 0.995, 0.1)
                if(done == 0):
                    if episode < 0:
                        self.ttt.showboard()
                        event = pygame.event.wait()
                        while event.type != pygame.MOUSEBUTTONDOWN:
                            event = pygame.event.wait()
                            if event.type == pygame.QUIT:
                                self.ttt.showboard()
                                print("pressed quit")
                                break
                        action=self.ttt.mouseClick()
                        self.ttt.showboard()
                    
                    
                    elif episode%3 == 0:
                        #_action = self.ttt.get_random_move()
                        _action = self.game.testQ(self.ttt.get_board(), self.possible_moves()) - 1
                        
                    else:
                        #_action = self.game.testQ(self.ttt.get_board(), self.possible_moves()) - 1
                        
                        reversed_state = self.ttt.get_bot_move()
                        
                        inp = np.array([reversed_state])
                        _preds = self.train_model.predict(inp)[0]
                        
                    
                        _action = self.get_highest_score_valid_move(reversed_state, _preds)
                        #action = np.argmax(_preds)
                        
                    if(action is not None):
                        reward, done = self.ttt.drawMove(_action, self.player)

                        '''
                        reversed_state = self.ttt.get_bot_move()
                        if(done == 1 or done == 2):
                            self.ER.store(pre_reversed_state, _action, reward * (-1), reversed_state)
                        else:
                            self.ER.store(pre_reversed_state, _action, reward, reversed_state)
                        '''
                       
                        
                
                
                if(done == 1):
                    winner += 1
                    
                elif(done == 2):
                    lost += 1
                    
                elif(done == 3):
                    draw += 1
                    
                total_step += 1
                if(done > 0):
                    total_rate += 1
                    win_result.append(winner/total_rate)
                    lost_result.append(lost/total_rate)
                    draw_result.append(draw/total_rate)
                
                next_state = self.ttt.get_state()
                
                self.ER.store(state, action, reward, next_state)
                replay = self.ER.get_random_minibatch(5)

                X = np.array(list(map(lambda x: x[0], replay)))
                y = []
                #X = []
                loss = 0
                for s, a, r, _s in replay:
                    #print(s)
                    #print(a)
                    #print('11111111111111')
                    
                    #if r != 1 or r != 0.5 or r != -1:
                    '''
                    if r == 0:
                        r = r + np.max(self.frozen_model.predict(np.array([_s]))) * self.gamma  #DQN
                    '''
                    _a = np.argmax(self.frozen_model.predict(np.array([_s])))
                    current = self.frozen_model.predict(np.array([s]))[0]
                    target = self.train_model.predict(np.array([_s]))[0] #DDQN
                    #print(target)
                    #print(_s)
                    
                    for point in range(9):
                        if(s[point+18] == 0):
                            target[point] = -10
                    
                    
                    
                    current[a] = r + self.gamma * target[_a]
                    y.append(current)
                    #X.append(s)
                    #X = np.array(X)
                y = np.array(y)

                if(len(X)>0):
                    self.train_model.train_on_batch(X, y)
                #loss += history[0]   
                #historys = self.train_model.fit(X, y, epochs=5, verbose=0)
                    #print(historys.history.keys())
                
                    #print('%.2f' % historys.history['acc'][0])
                state = next_state
                #print(loss)
                
                if(done != 0):
                    break
                
        print('total_step',total_step)
        print('winner',winner)
        print('lost',lost)
        print('draw',draw)
        print('greedy',greedy)
        plt.plot(win_result,color = 'red')
        plt.plot(lost_result, color = 'blue') 
        plt.plot(draw_result, color = 'black')    
        plt.show()       
        
        
        
        
        
    