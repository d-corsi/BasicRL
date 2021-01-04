import gym, os

class BasicRL:
	def __init__(self, algorithm, gym_env, verbose=1):
		self.algorithm = algorithm
		self.gym_env = gym_env
		self.verbose = verbose

		self.discrete_env = (type(self.gym_env.action_space) == gym.spaces.discrete.Discrete)

		valid_algorithms = ["REINFORCE", "ActorCritic", "A2C", "PPO", "mcPPO", "DDPG", "DQN", "TD3"]
		assert (algorithm in valid_algorithms), f"Invalid Algorithm! (options: {valid_algorithms})"

		self.change_default_paramters() #Reset all the parameters to default value
		if not os.path.exists("data"): os.makedirs("data") #Fix if folder does not exists


	def learn(self, ep_step):
		if self.algorithm == "REINFORCE": self._run_reinforce(ep_step)
		if self.algorithm == "ActorCritic": self._run_ActorCritic(ep_step)
		if self.algorithm == "A2C": self._run_A2C(ep_step)
		if self.algorithm == "PPO": self._run_PPO(ep_step)
		if self.algorithm == "mcPPO": self._run_mcPPO(ep_step)
		if self.algorithm == "DDPG": self._run_DDPG(ep_step)
		if self.algorithm == "DQN": self._run_DQN(ep_step)
		if self.algorithm == "TD3": self._run_TD3(ep_step)


	def change_default_paramters(self, gamma=None, sigma=None, memory_size=None, exploration_rate=None, exploration_decay=None, 
									batch_size=None, tau=None, noise_clip=None, actor_net=None, critic_net=None, epoch=None, render=None, save_model=False):
			self.gamma = gamma
			self.sigma = sigma
			self.memory_size = memory_size
			self.exploration_rate = exploration_rate
			self.exploration_decay = exploration_decay
			self.batch_size = batch_size
			self.tau = tau
			self.noise_clip = noise_clip
			self.actor_net = actor_net
			self.critic_net = critic_net
			self.epoch = epoch
			self.render = render
			self.save_model = save_model


	def _run_reinforce(self, ep_step):
		from BasicRL.REINFORCE import REINFORCE
		algorithm = REINFORCE( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_REINFORCE_model.h5")


	def _run_ActorCritic(self, ep_step):
		from BasicRL.ActorCritic import ActorCritic
		algorithm = ActorCritic( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_AC_model.h5")


	def _run_A2C(self, ep_step):
		from BasicRL.A2C import A2C
		algorithm = A2C( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_A2C_model.h5")


	def _run_PPO(self, ep_step):
		from BasicRL.PPO import PPO
		algorithm = PPO( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		if(self.batch_size != None): algorithm.batch_size = self.batch_size
		if(self.epoch != None): algorithm.epoch = self.epoch
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_PPO_model.h5")

	
	def _run_mcPPO(self, ep_step):
		from BasicRL.mcPPO import mcPPO
		algorithm = mcPPO( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		if(self.batch_size != None): algorithm.batch_size = self.batch_size
		if(self.epoch != None): algorithm.epoch = self.epoch
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_mcPPO_model.h5")


	def _run_DDPG(self, ep_step):
		from BasicRL.DDPG import DDPG
		assert (not self.discrete_env), "DDPG requires continuous environments!"
		algorithm = DDPG( self.gym_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.memory_size != None): algorithm.memory_size = self.memory_size
		if(self.exploration_rate != None): algorithm.exploration_rate = self.exploration_rate
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		if(self.batch_size != None): algorithm.batch_size = self.batch_size
		if(self.tau != None): algorithm.tau = self.tau
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_DDPG_model.h5")

	
	def _run_DQN(self, ep_step):
		from BasicRL.DQN import DQN
		assert (self.discrete_env), "DQN requires discrete environments!"
		algorithm = DQN( self.gym_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.memory_size != None): algorithm.memory_size = self.memory_size
		if(self.exploration_rate != None): algorithm.exploration_rate = self.exploration_rate
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		if(self.batch_size != None): algorithm.batch_size = self.batch_size
		if(self.tau != None): algorithm.tau = self.tau
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_DQN_model.h5")

	
	def _run_TD3(self, ep_step):
		from BasicRL.TD3 import TD3
		assert (not self.discrete_env), "TD3 requires continuous environments!"
		algorithm = TD3( self.gym_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.memory_size != None): algorithm.memory_size = self.memory_size
		if(self.exploration_rate != None): algorithm.exploration_rate = self.exploration_rate
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		if(self.batch_size != None): algorithm.batch_size = self.batch_size
		if(self.tau != None): algorithm.tau = self.tau
		if(self.noise_clip != None): algorithm.noise_clip = self.noise_clip
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_DDPG_model.h5")
		