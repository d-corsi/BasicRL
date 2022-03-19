from reinforcement_plotter import ReinforcementPlotter
import glob



data = [
		glob.glob("data/PPO_*.csv"),
		glob.glob("data/REINFORCE_*.csv"),
		glob.glob("data/mcPPO_*.csv"),
		glob.glob("data/DQN_*.csv"),
		glob.glob("data/DDPG_*.csv")
]

# Plot The Results
plotter = ReinforcementPlotter( x_label="episode", y_label="reward", title="Lunar Lander v2" )
plotter.load_array( data, key="reward", ref_line=0 )
plotter.process_data( rolling_window=100 )
plotter.render_std_log( labels=["PPO", "Reinforce", "PPO (Monte Carlo)", "DQN", "DDPG"], colors=["b", "g", "r", "c", "k"], styles=['-', '-', '-', '-', '-'] )
