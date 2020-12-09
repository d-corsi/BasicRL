import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from BasicRL.MyPlotter import MyPlotter

if __name__ == "__main__":
	import glob

	plotter = MyPlotter(x_label="Episode", y_label="Reward", title="None")
	plotter.load_array([
			glob.glob("data/reward_PPO_*.txt")
	])
	plotter.process_data( rolling_window=100, starting_pointer=30 )
	#plotter.render_std( labels=["Reinforce", "AC", "A2C", "PPO", "mcPPO", "DDPG"], colors=["b", "r", "g", "c", "y", "k"] )
	plotter.render_std( labels=["DQN", "mcPPO"], colors=["g", "r"] )
