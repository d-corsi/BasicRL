from basic_rl.main import BasicRL
import gym

if __name__ == "__main__":
	print("Hello Basic RL example!")

	seed = None
	env = gym.make( "LunarLanderContinuous-v2" ); env.seed( seed )
	basic_rl = BasicRL( "PPO", env, verbose=2, seed=seed )
	basic_rl.train( num_episode=5000 )
