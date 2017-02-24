import gym
env = gym.make('Pong-v0')

while True:
    env.render()
    action = int(input())
    env.step(action)
