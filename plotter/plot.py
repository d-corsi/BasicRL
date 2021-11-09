from my_plotter import MyPlotter
import glob

# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="reward", title="CartPole v1" )
plotter.load_array([
		glob.glob("data/reward_REINFORCE_*.txt"),
		glob.glob("data/reward_AC_*.txt"),
		glob.glob("data/reward_A2C_*.txt"),
		glob.glob("data/reward_PPO_*.txt"),
		glob.glob("data/reward_mcPPO_*.txt"),
		glob.glob("data/reward_DDQN_*.txt")
])
plotter.process_data( rolling_window=100, starting_pointer=0 )
plotter.render_std_log( labels=["reinforce", "actor-critic", "A2C", "PPO", "PPO (montecarlo)", "DDQN"], colors=["r", "g", "b", "y", "k", "c"] )
