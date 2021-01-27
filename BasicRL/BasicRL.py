import gym, os

class BasicRL:

	valid_algorithms = ["REINFORCE", "ActorCritic", "A2C", "PPO", "mcPPO", "DDPG", "DQN", "TD3"]

	def __init__(self, algorithm, gym_env, verbose=1, **kwargs ):

		# Class Attributes
		self.algorithm = algorithm
		self.gym_env = gym_env
		self.verbose = verbose
		
		# Class Attributes (optional)
		self.gamma = None
		self.sigma = None
		self.memory_size = None
		self.exploration_rate = None
		self.exploration_decay = None
		self.batch_size = None
		self.tau = None
		self.noise_clip = None
		self.actor_net = None
		self.critic_net = None
		self.epoch = None
		self.render = None
		self.save_model = False

		# Parse kwargs attribute
		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)

		# Private Variables
		self.discrete_env = (type(self.gym_env.action_space) == gym.spaces.discrete.Discrete)

		# Check valid algorithm
		assert (algorithm in self.valid_algorithms), f"Invalid Algorithm! (options: {self.valid_algorithms})"

		# Fix if folder does not exists
		if not os.path.exists("data"): os.makedirs("data") 


	def learn(self, ep_step):
		if self.algorithm == "REINFORCE": self._run_reinforce(ep_step)
		if self.algorithm == "ActorCritic": self._run_ActorCritic(ep_step)
		if self.algorithm == "A2C": self._run_A2C(ep_step)
		if self.algorithm == "PPO": self._run_PPO(ep_step)
		if self.algorithm == "mcPPO": self._run_mcPPO(ep_step)
		if self.algorithm == "DDPG": self._run_DDPG(ep_step)
		if self.algorithm == "DQN": self._run_DQN(ep_step)
		if self.algorithm == "TD3": self._run_TD3(ep_step)


	def _run_reinforce(self, ep_step):
		from BasicRL.algos.REINFORCE import REINFORCE
		algorithm = REINFORCE( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_REINFORCE_model.h5")


	def _run_ActorCritic(self, ep_step):
		from BasicRL.algos.ActorCritic import ActorCritic
		algorithm = ActorCritic( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_AC_model.h5")


	def _run_A2C(self, ep_step):
		from BasicRL.algos.A2C import A2C
		algorithm = A2C( self.gym_env, self.discrete_env, self.verbose )
		if(self.render != None): algorithm.render = self.render
		if(self.gamma != None): algorithm.gamma = self.gamma
		if(self.sigma != None): algorithm.sigma = self.sigma
		if(self.exploration_decay != None): algorithm.exploration_decay = self.exploration_decay
		algorithm.loop(ep_step)

		if(self.save_model): algorithm.actor.save("data/final_A2C_model.h5")


	def _run_PPO(self, ep_step):
		from BasicRL.algos.PPO import PPO
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
		from BasicRL.algos.mcPPO import mcPPO
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
		from BasicRL.algos.DDPG import DDPG
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
		from BasicRL.algos.DQN import DQN
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
		from BasicRL.algos.TD3 import TD3
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
		