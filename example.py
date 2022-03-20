from basic_rl.main import BasicRL
import gym


# Example class to launch BasicRL on a simple Gym environment
# Optionally can provide a roandom seed for reproducibility
def main( seed=None ):

	# Create the environment, the BasicRL object and run the training
	env = gym.make( "LunarLanderContinuous-v2" ); env.seed( seed )
	basic_rl = BasicRL( "PPO", env, verbose=2, seed=seed )
	basic_rl.train( num_episode=5000 )


# "python ./example.py" to launch the example
if __name__ == "__main__":
	print("Hello Basic RL example!")
	main()
