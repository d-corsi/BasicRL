from my_plotter import MyPlotter
import glob

# Plot The Results
plotter = MyPlotter( x_label="episode", y_label="reward", title="Continuous Lunar Lander v2" )
plotter.load_array([
		glob.glob("data/reward_REINFORCE_*.txt"),
		glob.glob("data/reward_A2C_*.txt"),
		glob.glob("data/reward_PPO_*.txt")
])
plotter.process_data( rolling_window=300, starting_pointer=0 )
plotter.render_std( labels=["reinforce", "A2C", "PPO"], colors=["m", "g", "c"] )
