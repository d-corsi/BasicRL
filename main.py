import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from BasicRL.BasicRL import BasicRL
import gym; gym.logger.set_level(40)


if __name__ == "__main__":
	print("Hello Basic RL!")

	#env = gym.make("CartPole-v1")
	#env = gym.make("LunarLanderContinuous-v2")
	env = gym.make("LunarLander-v2")

	learner = BasicRL("DQN", gym_env=env, verbose=2)
	learner.change_default_paramters(gamma=0.99, memory_size=10000, batch_size=128, exploration_rate=1, exploration_decay=0.995, tau=0.005)
	learner.learn(1000)


# DQN
#learner.change_default_paramters(gamma=0.99, memory_size=5000, batch_size=64, exploration_rate=1, exploration_decay=0.995, tau=0.005)
# PPO
#learner.change_default_paramters(gamma=0.99, sigma=1.0, exploration_decay=0.995, batch_size=128, epoch=10)