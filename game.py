import numpy as np
from os import system

class State():
    
    def __init__(self):
        self.actions = []
        self.reward = -0.1
        self.agent = False
        self.terminal = False
        
class Map():
    
    def __init__(self):
        all_states = [State() for i in range(36)]
        all_states[0].actions = ['D', 'R']
        all_states[5].actions = ['D', 'L']
        all_states[35].actions = ['U', 'L']
        all_states[30].actions = ['U', 'R']
        for i in range(36):
            if i not in [0,1,2,3,4,5,30,31,32,33,34,35,6,12,18,24,30,11,17,23,29]:
                all_states[i].actions = ['U', 'D', 'L', 'R']
        for i in [1,2,3,4]:
            all_states[i].actions = ['L', 'D', 'R']
        for i in [6,12,18,24]:
            all_states[i].actions = ['U', 'D', 'R']
        for i in [11,17,23,29]:
            all_states[i].actions = ['U', 'D', 'L']
        for i in range(31, 35):
            all_states[i].actions = ['L', 'U', 'R']
        self.all_states = all_states
        self.all_states[6].reward = -1
        self.all_states[6].terminal = True
        self.all_states[8].reward = -1
        self.all_states[8].terminal = True
        self.all_states[10].reward = -1
        self.all_states[10].terminal = True
        self.all_states[19].reward = -1
        self.all_states[19].terminal = True
        self.all_states[21].reward = -1
        self.all_states[21].terminal = True
        self.all_states[29].reward = -1
        self.all_states[29].terminal = True
        self.all_states[30].reward = -1
        self.all_states[30].terminal = True
        self.all_states[32].reward = -1
        self.all_states[32].terminal = True
        self.all_states[16].reward = 1
        self.all_states[16].terminal = True
        self.all_states[31].agent = True
        for i in range(36):
            all_states[i].token = i
    
    def displayMap(self):
        
        system('clear')
        row1 = np.array(['-','-','-','-','-','-'])
        row2 = np.array(['X','-','X','-','X','-'])
        row3 = np.array(['-','-','-','-','$','-'])
        row4 = np.array(['-','X','-','X','-','-'])
        row5 = np.array(['-','-','-','-','-','X'])
        row6 = np.array(['X','-','X','-','-','-'])
        rows = [row1, row2, row3, row4, row5, row6]
        for state in self.all_states:
            if state.agent == True:
                pos = self.all_states.index(state)
        row = pos//6
        col = pos%6
        rows[row][col] = 'o'
        print(row1)
        print(row2)
        print(row3)
        print(row4)
        print(row5)
        print(row6)
        
        
    def currentState(self):
        for state in self.all_states:
            if state.agent == True:
                return state
        
    def setState(self, statep):
        state = self.currentState()
        state.agent = False
        statep.agent = True
        return statep.reward
        
def move(map1, move):
    state = map1.currentState()
    if move in state.actions:
        if move == 'U':
            statep = map1.all_states[map1.all_states.index(state)-6]
        if move == 'L':
            statep = map1.all_states[map1.all_states.index(state)-1]
        if move == 'D':
            statep = map1.all_states[map1.all_states.index(state)+6]
        if move == 'R':
            statep = map1.all_states[map1.all_states.index(state)+1]
        reward = map1.setState(statep)
        return reward
    return state.reward
        
def gameOver(map1):
    state = map1.currentState()
    if state.terminal == True:
        return True


def standardMap():
    map1 = Map()
    return map1

map1 = standardMap()
map1.displayMap()
while True:
    mv = input()
    move(map1, mv)
    map1.displayMap()
    if gameOver(map1):
        break
map1.displayMap()
