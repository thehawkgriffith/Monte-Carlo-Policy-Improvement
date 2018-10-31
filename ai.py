from game import standardMap
from game import move
from game import gameOver
from time import sleep
import numpy as np

map1 = standardMap()


def max_dict(d):
	max_key = None
	max_value = float('-inf')
	for k, v in d.items():
		if v > max_value:
			max_value = v
			max_key = k
	return max_key, max_value

class Agent():

	def __init__(self):
		self.memory = {}
		self.policy = {}
		self.Q = {}


	def playAnEpisode(self, map1):
		states = []
		for state in map1.all_states:
			if state.terminal == False:
				states.append(state)
		s = np.random.choice(states)
		a = np.random.choice(s.actions)
		map1.setState(s)
		states_actions_rewards = [(s.token, a, 0)]
		seen_states = set()
		seen_states.add(map1.currentState().token)
		num_steps = 0
		while True:
			r = move(map1, a)
			num_steps += 1
			s = map1.currentState()

			if s.token in seen_states:
				reward = -10./num_steps
				states_actions_rewards.append((s.token, None, reward))
				break

			elif gameOver(map1):
				states_actions_rewards.append((s.token, None, r))
				break
			
			else:
				a = self.policy[s.token]
				states_actions_rewards.append((s.token, a, r))
			seen_states.add(s.token)

		G = 0
		states_actions_returns = []
		first = True
		for s, a, r in reversed(states_actions_rewards):
			if first:
				first = False
			else:
				states_actions_returns.append((s, a, G))
			G = r + (0.9 * G)
		states_actions_returns.reverse()
		return states_actions_returns

	def generateOptimalPolicy(self, map1):
		states = []
		for state in map1.all_states:
			if state.terminal == False:
				states.append(state)
		for s in states:
			self.policy[s.token] = np.random.choice(s.actions)
		self.returns = {}
		for s in map1.all_states:
			if s.terminal == False:
				self.Q[s.token] = {}
				for a in s.actions:
					self.Q[s.token][a] = 0
					self.returns[(s.token, a)] = []
			else:
				pass

		for T in range(2000):
			self.states_actions_returns = self.playAnEpisode(map1)
			seen_s_a_pairs = set()
			for s, a, G in self.states_actions_returns:
				if (s, a) not in seen_s_a_pairs:
					old_q = self.Q[s][a]
					self.returns[(s, a)].append(G)
					self.Q[s][a] = max(self.returns[(s, a)])
					seen_s_a_pairs.add((s, a))

		for sa in states:
			self.policy[sa.token] = max_dict(self.Q[sa.token])[0]

agent = Agent()
agent.generateOptimalPolicy(map1)
map1 = standardMap()
map1.setState(map1.all_states[13])
map1.displayMap()
while True:
    mv = agent.policy[map1.currentState().token]
    move(map1, mv)
    map1.displayMap()
    sleep(0.3)
    if gameOver(map1):
        break
map1.displayMap()