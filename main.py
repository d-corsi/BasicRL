import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from BasicRL.BasicRL import BasicRL
import gym; gym.logger.set_level(40)
import safety_gym, panda_gym


if __name__ == "__main__":
	print("Hello Basic RL!")

	#env = gym.make("CartPole-v1")
	#env = gym.make("LunarLanderContinuous-v2")
	#env = gym.make("LunarLander-v2")
	#env = gym.make('Safexp-PointPush1-v0')
	env = gym.make('PandaReach-v0', render=True)

	learner = BasicRL("PPO", gym_env=env, verbose=1)
	learner.change_default_paramters(gamma=0.99, sigma=1.0, exploration_decay=0.99)
	learner.learn(150)


# PPO: learner.change_default_paramters(gamma=0.99, sigma=1.0, exploration_decay=0.99, batch_size=128, epoch=10)
# DQN: learner.change_default_paramters( gamma=0.99, memory_size=10000, exploration_decay=0.99, batch_size=128 )