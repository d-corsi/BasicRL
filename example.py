from BasicRL.BasicRL import BasicRL
from BasicRL.MyPlotter import MyPlotter
import gym, glob

if __name__ == "__main__":
	print("Hello Basic RL example!")
	
	# Load Gym Env
	env = gym.make("CartPole-v1")

	# Run PPO Algorithm
	learner = BasicRL("PPO", gym_env=env, verbose=2, gamma=0.99, sigma=1.0, exploration_decay=0.99)
	learner.learn(300)
	
	# Run DQN Algorithm
	learner = BasicRL("DQN", gym_env=env, verbose=2, gamma=0.99, memory_size=10000, exploration_decay=0.99, batch_size=128)
	learner.learn(300)
	
	# Plot The Results
	plotter = MyPlotter(x_label="Episode", y_label="Reward", title="CartPole v1")
	plotter.load_array([
			glob.glob("data/reward_PPO_*.txt"),
			glob.glob("data/reward_DQN_*.txt")
	])
	plotter.process_data( rolling_window=100, starting_pointer=30 )
	plotter.render_std( labels=["PPO", "DQN"], colors=["g", "r"] )
	