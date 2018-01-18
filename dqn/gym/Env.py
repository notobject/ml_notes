# encoding=utf-8
# Created by Mr.Long on 2017/12/29 0029.
# 这是文件的概括

import pygame, sys
import numpy
import random
import cv2


class Env:
    def __init__(self, env_name):
        self.width = 320
        self.height = 240
        self.image_width = 120
        self.image_height = 80
        self.dim = 1
        self.block_size = 10
        self.block_speed = 8
        self.block_x = 0
        self.block_y = 0
        self.block_direction_x = 1
        self.block_direction_y = 1
        self.bridge_width = 50
        self.bridge_height = 5
        self.bridge_speed = 20
        self.bridge_x = 0
        self.bridge_y = 0
        pygame.init()
        pygame.display.set_caption(env_name)
        self.screen = pygame.display.set_mode([self.width, self.height], 0, 8)
        pass

    def current_state(self):
        block_center_x = self.block_x + self.block_size / 2
        block_center_y = self.block_y + self.block_size / 2
        bridge_center_x = self.bridge_x + self.bridge_width / 2
        bridge_center_y = self.bridge_y + self.bridge_height / 2
        return [block_center_x, block_center_y, bridge_center_x, bridge_center_y,self.block_direction_x,self.block_direction_y]

    def state_dim(self):
        return 6

    def action_dim(self):
        return 3

    def reset(self):
        self.block_x = numpy.random.randint(0, self.width - self.block_size)
        self.block_y = 0
        self.block_direction_x = 1 if numpy.random.randint(0, 2) == 0 else -1
        self.block_direction_y = 1
        self.bridge_x = self.width / 2 - self.bridge_width / 2
        self.bridge_y = 0
        self.redraw()
        return self.current_state()

    def get_block_rect(self):
        return pygame.rect.Rect([self.block_x, self.block_y, self.block_size, self.block_size])

    def get_bridge_rect(self):
        return pygame.rect.Rect(
            [self.bridge_x, self.height - self.bridge_height, self.bridge_width, self.bridge_height])

    def redraw(self):
        self.screen.fill([0, 0, 0])
        self.screen.fill(color=[255, 255, 255], rect=self.get_block_rect())
        self.screen.fill(color=[255, 255, 255], rect=self.get_bridge_rect())
        pygame.display.update()
        for event in pygame.event.get():
            pass
        pass

    def step(self, action, delay=0):
        # next_state, reward, done, _
        self.block_x += self.block_speed * self.block_direction_x
        self.block_y += self.block_speed * self.block_direction_y

        if self.block_x + self.block_size >= self.width:
            self.block_direction_x = -1
        elif self.block_y <= 0:
            self.block_direction_y = 1
        elif self.block_x <= 0:
            self.block_direction_x = 1
        if action == 0:
            pass
        elif action == 1:
            self.bridge_x -= self.bridge_speed
            if self.bridge_x < 0:
                self.bridge_x = 0
        elif action == 2:
            self.bridge_x += self.bridge_speed
            if self.bridge_x >= self.width - self.bridge_width:
                self.bridge_x = self.width - self.bridge_width

        block_center_x = self.block_x + self.block_size / 2
        block_center_y = self.block_y + self.block_size / 2
        bridge_center_x = self.bridge_x + self.bridge_width / 2
        bridge_center_y = self.bridge_y + self.bridge_height / 2

        reward = 300 - numpy.sqrt(numpy.sum(numpy.square(bridge_center_x - block_center_x)))
        done = False
        over = False
        if self.get_block_rect().colliderect(self.get_bridge_rect()):
            over = True
            done = True
            reward += 1000
        elif self.block_y > self.height - self.block_size:
            over = True
            done = False
            reward -= 3000
        self.redraw()
        pygame.time.delay(delay)
        return self.current_state(), reward, done, over

    def set_step(self, episode):
        pygame.display.set_caption("%s - %d" % (ENV_NAME, episode))
        pass


ENV_NAME = 'DADADA'
EPISODE = 100000
STEP = 29
TEST = 10


def test():
    total_reward = 0
    error_count = 0
    test_count = 0
    for i in range(TEST):
        state = env.reset()
        actions = []
        for j in range(STEP):
            # env.render()
            action = agent.action(state)  # direct action for test
            actions.append(action)
            state, reward, done, over = env.step(action, delay=10)
            total_reward += reward
            if over:
                test_count += 1
                if not done:
                    error_count += 1
                    break
                break

    acc = 1 - (error_count / test_count)
    avg_reward = total_reward / TEST
    print('episode: ', episode, 'Evaluation Acc Rate:%g, AverageReward: %g' % (acc, avg_reward))


if __name__ == "__main__":

    import DQN as dqn

    env = Env(ENV_NAME)
    agent = dqn.DQN(env)
    for episode in range(EPISODE):
        # initialize task
        env.set_step(episode)
        state = env.reset()
        # Train
        for step in range(STEP):
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action, delay=5)
            # Define reward for agent
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if episode % 100 == 0:
            test()
