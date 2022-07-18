from reinforcement_plotter import ReinforcementPlotter
import glob


# Collect the data from the CSV file
data = [
		glob.glob("log/PPO_*/run_stats.csv"),
		#glob.glob("log/REINFORCE_*/run_stats.csv"),
		#glob.glob("log/mcPPO_*/run_stats.csv"),
		#glob.glob("log/DQN_*/run_stats.csv"),
		#glob.glob("log/DDPG_*/run_stats.csv")
]

# Plot The Results
plotter = ReinforcementPlotter( x_label="episode", y_label="reward", title="Lunar Lander v2" )
plotter.load_array( data, key="reward", ref_line=0 )
plotter.process_data( rolling_window=100 )
plotter.render_std_log( labels=["PPO", "Reinforce", "PPO (Monte Carlo)", "DQN", "DDPG"], colors=["b", "g", "r", "c", "k"], styles=['-', '-', '-', '-', '-'] )
