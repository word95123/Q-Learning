import pygame
import random
import time
import numpy as np




class TicTacToes:
    def __init__(self,traning=False):
        self.board = [' ']*9
        self.fmove = None
        self.done = False
        self.humman=None
        self.computer=None
        self.humanTurn=None
        self.training=traning
        self.player1 = None
        self.player2 = None
        self.aiplayer=None
        self.isAI=False
        self.player1win = None
        self.player2win = None
        self.state = None


        # if not training display
        if(not self.training):
            pygame.init()
            self.ttt = pygame.display.set_mode((225,250))
            pygame.display.set_caption('Tic-Tac-Toe')

    
    #reset the game
    def reset(self):
        self.fmove = int(random.random()*2)
        self.state = [0] * 18 + [1] * 9
        self.board = [' '] * 9
        self.humanTurn=random.choice([True,False])
        legal_moves = [moves for moves, v in enumerate(self.board) if v == ' ']

        self.surface = pygame.Surface(self.ttt.get_size())
        self.surface = self.surface.convert()
        self.surface.fill((250, 250, 250))
        #horizontal line
        pygame.draw.line(self.surface, (0, 0, 0), (75, 0), (75, 225), 2)
        pygame.draw.line(self.surface, (0, 0, 0), (150, 0), (150, 225), 2)
        # veritical line
        pygame.draw.line(self.surface, (0, 0, 0), (0,75), (225, 75), 2)
        pygame.draw.line(self.surface, (0, 0, 0), (0,150), (225, 150), 2)
        
        return self.state, legal_moves

   #evaluate function
    def evaluate(self, ch):
        # "rows checking"
        for i in range(3):
            if (ch == self.board[i * 3] == self.board[i * 3 + 1] and self.board[i * 3 + 1] == self.board[i * 3 + 2]):
                return True
        # "col checking"
        for i in range(3):
            if (ch == self.board[i + 0] == self.board[i + 3] and self.board[i + 3] == self.board[i + 6]):
                return True
        # diagonal checking
        if (ch == self.board[0] == self.board[4] and self.board[4] == self.board[8]):
            return True

        if (ch == self.board[2] == self.board[4] and self.board[4] == self.board[6]):
            return True
        # "if filled draw"
        #if not any(c == ' ' for c in self.board):
            #return True

        return False

    def board_is_full(self):
        full_state = 0
        for i in range(int(len(self.board)*2)):
            full_state += self.state[i]
        if(full_state == 9):
            return True
        return False
    #return remaining possible moves
    def possible_moves(self):
        return [moves for moves, v in enumerate(self.board) if v == ' ']

    #take next step and return reward
    def step(self, action, move):
        done = 0
        if self.board[action] != ' ':
            return -1
        self.board[action] = move
        win = self.evaluate(move)
        if(move == 'X'):
            if(win):
                done = 1 #botWin
            elif(self.board_is_full()):
                done = 3 #draw
        else:
            if(win):
                done = 2 #playerWin
            elif(self.board_is_full()):
                done = 3 #draw

        return done
    
    def get_state(self):
        return self.state
    
    def get_board(self):
        return self.board
    
    def get_random_move(self):
        space_free = [moves for moves, v in enumerate(self.board) if v == ' ']
        return random.choice(space_free)
    def get_bot_move(self):
        reversed_state = [0] * 18 + [1] * 9
        #reversed_state = np.array(reversed_state)
        for i in range(len(self.state)):
            if(i < len(self.board)):
                reversed_state[i+9] = self.state[i]
            elif(i < int(len(self.board) * 2)):
                reversed_state[i-9] = self.state[i]
            else:
                reversed_state[i] = self.state[i]
        return reversed_state

    #draw move on window
    def drawMove(self, action, move):
        
        
        if(move == 'X'):
            self.state[action] = 1
        else:
            self.state[9 + action] = 1
        self.state[18 + action] = 0
        
        row=int((action)/3)
        col=(action)%3

        centerX = ((col) * 75) + 32
        centerY = ((row) * 75) + 32

        done = self.step(action, move)
        reward = 0
        if(done == -1): #illegal move
            reward = -2
        elif(done == 2): #loss
            reward = -10
        elif(done == 1): #win
            reward = 10
        elif(done == 3): #draw
            reward = 0.5
        
        
        if(reward == -2): #overlap
            #print('Invalid move')
            font = pygame.font.Font(None, 24)
            text = font.render('Invalid move!', 1, (10, 10, 10))
            self.surface.fill((250, 250, 250), (0, 300, 300, 25))
            self.surface.blit(text, (10, 230))
            reward = -1

            return reward, done

        if (move == 'X'): #playerX so draw x
            font = pygame.font.Font(None, 24)
            text = font.render('X', 1, (10, 10, 10))
            self.surface.fill((250, 250, 250), (0, 300, 300, 25))
            self.surface.blit(text, (centerX, centerY))
            self.board[action] ='X'

            if(done == 1): #if playerX is humman and won, display humman won
                #print('Humman won! in X')
                text = font.render('Computer won!', 1, (10, 10, 10))
                self.surface.fill((250, 250, 250), (0, 300, 300, 25))
                self.surface.blit(text, (10, 230))

        else:  #playerO so draw O
            font = pygame.font.Font(None, 24)
            text = font.render('O', 1, (10, 10, 10))

            self.surface.fill((250, 250, 250), (0, 300, 300, 25))
            self.surface.blit(text, (centerX, centerY))
            self.board[action] = 'O'

            if (done == 2):  #if playerO is computer and won, display computer won
                #print('computer won! in O')
                text = font.render('Human won!', 1, (10, 10, 10))
                self.surface.fill((250, 250, 250), (0, 300, 300, 25))
                self.surface.blit(text, (10, 230))



        if (done == 3):  # draw, then display draw
            #print('Draw Game! in O')
            font = pygame.font.Font(None, 24)
            text = font.render('Draw Game!', 1, (10, 10, 10))
            self.surface.fill((250, 250, 250), (0, 300, 300, 25))
            self.surface.blit(text, (10, 230))
            return reward, done

        return reward, done

    # mouseClick position
    def mouseClick(self):
        (mouseX, mouseY) = pygame.mouse.get_pos()
        if (mouseY < 75):
            row = 0
        elif (mouseY < 150):
            row = 1
        else:
            row = 2

        if (mouseX < 75):
            col = 0
        elif (mouseX < 150):
            col = 1
        else:
            col = 2
        return row * 3 + col 




    #show display
    def showboard(self):
        self.ttt.blit(self.surface, (0, 0))
        pygame.display.flip()



            