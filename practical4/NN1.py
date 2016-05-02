import numpy as np
from SwingyMonkey import SwingyMonkey
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import sgd
import numpy.random as npr
import random
from stub import Learner as L
from random import sample as rsample
from keras.layers.convolutional import Convolution1D

def experience_replay(batch_size):
    """
    Coroutine of experience replay.
    
    Provide a new experience by calling send, which in turn yields 
    a random batch of previous replay experiences.
    """
    memory = []
    while True:
        experience = yield rsample(memory, batch_size) if batch_size <= len(memory) else None
        memory.append(experience)


# class ExperienceReplay(object):
#     def __init__(self, max_memory=100, discount=.9):
#         self.max_memory = max_memory
#         self.memory = list()
#         self.discount = discount

#     def remember(self, states):
#         # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
#         self.memory.append([states])
#         if len(self.memory) > self.max_memory:
#             del self.memory[0]

#     def get_batch(self, model, batch_size=10):
#         len_memory = len(self.memory)
#         num_actions = model.output_shape[-1]
#         env_dim = self.memory[0][0][0].shape[1]
#         inputs = np.zeros((min(len_memory, batch_size), env_dim))
#         targets = np.zeros((inputs.shape[0], num_actions))
#         for i, idx in enumerate(np.random.randint(0, len_memory,
#                                                   size=inputs.shape[0])):
#             state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]

#             inputs[i:i+1] = state_t
#             # There should be no target values for actions not taken.
#             # Thou shalt not correct actions not taken #deep
#             targets[i] = model.predict(state_t)[0]
#             Q_sa = np.max(model.predict(state_tp1)[0])
#             if reward_t < 0:  # if game_over is True
#                 targets[i, action_t] = reward_t
#             else:
#                 # reward_t + gamma * max_a' Q(s', a')
#                 targets[i, action_t] = reward_t + self.discount * Q_sa
#         return inputs, targets


class NN(L):
    def __init__(self):
        self.last_state=None
        self.last_action=None
        self.last_reward=None

        self.aa = np.zeros((4,2))

        hidden_size = 100
        self.batch_size = 10

        self.gamma = 0.9
        self.loss = 0.
        self.win_cnt = 0
            
        self.epsilon = 0.9  # exploration
        self.num_actions = 2  # 
        
        self.model=Sequential()
        
        self.model.add(Dense(128,input_shape=(210,)))
        self.model.add(Activation('relu'))
        
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))


        self.model.add(Dense(self.num_actions))
        self.model.add(Activation('relu'))
        
        rms=RMSprop()
        self.model.compile(loss='mse',optimizer=rms)

        

    def action_callback(self,state):


        input_t = self.con_state(state)
        input_tm1 = self.con_state(self.last_state)

        #print input_t,input_tm1

        model=self.model
        


        if np.random.rand() <= self.epsilon:
            action = 1*(np.random.rand() < 0.3)
        else:
            q = model.predict(input_t[np.newaxis])
            action = np.argmax(q[0])
        # if action ==1:
        #     print action

        if self.last_state != None:

        #     t = model.predict(input_tm1).flatten()
        #     ix = self.last_action
        #     if self.last_reward < 0:
        #         t[ix] = self.last_reward
        #     else:
        #         t[ix] = self.last_reward + self.gamma * model.predict(input_t).max(axis=-1)

        #     model.fit(input_tm1,t.reshape(1,2),batch_size=1,nb_epoch=1,verbose=0)

            experience = (input_tm1, self.last_action, self.last_reward, input_t)
            
            batch = exp_replay.send(experience)
            if batch:
                inputs = []
                targets = []
                for s, a, r, s_prime in batch:
                    # The targets of unchosen actions are set to the q-values of the model,
                    # so that the corresponding errors are 0. The targets of chosen actions
                    # are set to either the rewards, in case a terminal state has been reached, 
                    # or future discounted q-values, in case episodes are still running.
                    t = model.predict(s[np.newaxis]).flatten()
                    ix = a
                    if r < 0:
                        t[ix] = r
                    else:
                        t[ix] = r + self.gamma * model.predict(s_prime[np.newaxis]).max(axis=-1)
                    targets.append(t)
                    inputs.append(s)
                self.loss += model.train_on_batch(np.array(inputs), np.array(targets))[0]        
        # store experience
        # if self.last_state != None:

        #     exp_replay.remember([input_tm1, self.last_action, self.last_reward, input_t])            

        #     # adapt model
        #     inputs, targets = exp_replay.get_batch(model, batch_size=self.batch_size)

        #     self.loss += model.train_on_batch(inputs, targets)[0]
            #model.fit(inputs,targets,batch_size=self.batch_size,verbose=1)

        if self.last_reward == 1:
            self.win_cnt += 1
        #print "Loss",loss

        self.last_action = action
        self.last_state = state

        return action
        

    def reward_callback(self,reward):
        self.last_reward=reward

    #take state return four value, with normalized value
    def con_state(self,state):
        
        if state == None:
            return None

        tdist=state['tree']['dist']
        ttop=state['tree']['top']
        tbot=state['tree']['bot']
        mvel=state['monkey']['vel']
        mtop=state['monkey']['top']
        mbot=state['monkey']['bot']

        # if mvel > self.aa[0,0]:
        #     self.aa[0,0] = mvel
        # if mvel < self.aa[0,1]:
        #     self.aa[0,1] = mvel
        # if (ttop-mtop) > self.aa[1,0]:
        #     self.aa[1,0] = ttop-mtop
        # if (ttop-mtop) < self.aa[1,1]:
        #     self.aa[1,1] = ttop-mtop

        # if tdist > self.aa[3,0]:
        #     self.aa[3,0] = tdist
        # if tdist < self.aa[3,1]:
        #     self.aa[3,1] = tdist
        #print self.aa
        

        v = (mvel + 60) / 10 
        h = (ttop-mtop+209) / 6
        w = (tdist+120) / 7

        matrix = np.zeros(210)

        matrix[v] = 1
        matrix[h+11] = 1
        matrix[w+111] = 1

        return matrix

    def print_s(self):
        if self.epsilon > .1:
            self.epsilon -= .9 / (10000 / 2.0)

        print("Loss {:.4f} | Win count {}".format(np.mean(self.loss), self.win_cnt))

    def reset(self):
        self.loss = 0.
        self.win_cnt = 0



def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        

        print ii,

        learner.print_s()
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

    # Select agent.
    agent = NN()

    # Empty list to save history.
    hist = []
    max_memory = 500
    batch_size = 10

    exp_replay = experience_replay(batch_size)
    exp_replay.next()  # Start experience replay coroutine


    # exp_replay = ExperienceReplay(max_memory=max_memory)

    # Run games. 
    run_games(agent, hist, 15000, 1)

    # Save history. 
    np.save('hist',np.array(hist))
