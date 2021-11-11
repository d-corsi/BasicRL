from basic_rl.main import BasicRL
import gym

if __name__ == "__main__":
	print("Hello Basic RL example!")
	
	env = gym.make("CartPole-v1")
	basic_rl = BasicRL( "PPO", env, verbose=1 )
	basic_rl.train()
