from DGame import TicTacToes
#from QLearning import  Qlearning
import pygame
import numpy as np
import time
import random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

#game = TicTacToe() #game instance
#player1=Humanplayer() #human player
#player2=Qlearning()  #agent

ttt = TicTacToes()
player = 'O'
ai = 'X'


frozen_model = Sequential()
frozen_model.add(Dense(units=243,input_dim=27, activation = 'relu'))
frozen_model.add(Dense(units=243, activation = 'relu'))
frozen_model.add(Dense(units=9, activation = 'linear'))
frozen_model.load_weights('weights.h5')

def get_highest_score_valid_move(state, action):
    s = []
    for i in range(int(len(state)/2)):
        s.append(state[i] + state[i+9])
    for move_choice in action.argsort()[::-1]:
        if s[move_choice] == 0:
            return move_choice
'''
def get_highest_score_valid_move(self, state, action):
    s = []
    for i in range(int(len(state)/2)):
        if (state[i] + state[i+9]) !=0:
            action[i] = -1
 
    return action
'''   
def startGames():
    state, legal_moves = ttt.reset()
    while(True):
        who_first = int(random.random()*2)
        print(who_first)
        if(who_first == 1):
            while(True):
                inp = np.array([state])
                print(inp)
                preds = frozen_model.predict(inp)[0]
                print(preds)
                space_free = [moves for moves, v in enumerate(ttt.get_board()) if v != ' ']
                '''
                for i in space_free:
                    preds[i] = -10
                '''
                action = get_highest_score_valid_move(state, preds)
                #action = np.argmax(preds)
                
                reward, done = ttt.drawMove(action, ai)
                #state = ttt.get_state()
                ttt.showboard()

                if (done!=0): #if done reset
                    time.sleep(1)
                    state, legal_moves = ttt.reset()
                    break
                event = pygame.event.wait()
                while event.type != pygame.MOUSEBUTTONDOWN:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT:
                        ttt.showboard()
                        print("pressed quit")
                        break
                action=ttt.mouseClick()
                reward, done = ttt.drawMove(action, player)

                ttt.showboard()

                if (done!=0): #if done reset
                    time.sleep(3)
                    state, legal_moves = ttt.reset()
                    break
                
        else:
            while(True):
                ttt.showboard()
                event = pygame.event.wait()
                while event.type != pygame.MOUSEBUTTONDOWN:
                    event = pygame.event.wait()
                    if event.type == pygame.QUIT:
                        ttt.showboard()
                        print("pressed quit")
                        break
                action=ttt.mouseClick()
                reward, done = ttt.drawMove(action, player)

                ttt.showboard()

                if (done!=0): #if done reset
                    time.sleep(1)
                    state, legal_moves = ttt.reset()
                    break
                inp = np.array([state])
                print(inp)
                preds = frozen_model.predict(inp)[0]
                print(preds)
                space_free = [moves for moves, v in enumerate(ttt.get_board()) if v != ' ']
                '''
                for i in space_free:
                    preds[i] = -10
                '''
                action = get_highest_score_valid_move(state, preds)
                #action = np.argmax(preds)
                
                reward, done = ttt.drawMove(action, ai)
                #state = ttt.get_state()
                ttt.showboard()



                if (done!=0): #if done reset
                    time.sleep(3)
                    state, legal_moves = ttt.reset()
                    break

startGames()
#game.startGame(player1,player2)#player1 is X, player2 is 0
#game.reset() #reset
#game.render(training=False) # render display