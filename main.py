import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from BasicRL.BasicRL import BasicRL
import safety_gym
import gym; gym.logger.set_level(40)

if __name__ == "__main__":
	print("Hello Basic RL!")

	#env = gym.make("CartPole-v1")
	#env = gym.make("LunarLanderContinuous-v2")
	#env = gym.make("LunarLander-v2")
	env = gym.make('Safexp-PointGoal1-v0')

	learner = BasicRL("mcPPO", gym_env=env, verbose=2)
	learner.change_default_paramters(gamma=0.99, sigma=1.0, exploration_decay=0.99, batch_size=128, epoch=10)
	learner.learn(1000)
