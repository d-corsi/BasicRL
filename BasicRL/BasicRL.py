import gym

class BasicRL:
	def __init__(self, algorithm, gym_env, verbose=1):
		self.algorithm = algorithm
		self.gym_env = gym_env
		self.verbose = verbose

		self.discrete_env = (type(self.gym_env.action_space) == gym.spaces.discrete.Discrete)

		valid_algorithms = ["REINFORCE", "ActorCritic", "A2C", "PPO", "mcPPO", "DDPG", "DQN"]
		assert (algorithm in valid_algorithms), f"Invalid Algorithm! (options: {valid_algorithms})"


	def learn(self, ep_step):
		if self.algorithm == "REINFORCE": self._run_reinforce(ep_step)
		if self.algorithm == "ActorCritic": self._run_ActorCritic(ep_step)
		if self.algorithm == "A2C": self._run_A2C(ep_step)
		if self.algorithm == "PPO": self._run_PPO(ep_step)
		if self.algorithm == "mcPPO": self._run_mcPPO(ep_step)
		if self.algorithm == "DDPG": self._run_DDPG(ep_step)
		if self.algorithm == "DQN": self._run_DQN(ep_step)

	def _run_reinforce(self, ep_step):
		from BasicRL.REINFORCE import REINFORCE
		algorithm = REINFORCE( self.gym_env, self.discrete_env, self.verbose )
		algorithm.loop(ep_step)


	def _run_ActorCritic(self, ep_step):
		from BasicRL.ActorCritic import ActorCritic
		algorithm = ActorCritic( self.gym_env, self.discrete_env, self.verbose )
		algorithm.loop(ep_step)


	def _run_A2C(self, ep_step):
		from BasicRL.A2C import A2C
		algorithm = A2C( self.gym_env, self.discrete_env, self.verbose )
		algorithm.loop(ep_step)


	def _run_PPO(self, ep_step):
		from BasicRL.PPO import PPO
		algorithm = PPO( self.gym_env, self.discrete_env, self.verbose )
		algorithm.loop(ep_step)

	
	def _run_mcPPO(self, ep_step):
		from BasicRL.mcPPO import mcPPO
		algorithm = mcPPO( self.gym_env, self.discrete_env, self.verbose )
		algorithm.loop(ep_step)


	def _run_DDPG(self, ep_step):
		from BasicRL.DDPG import DDPG
		assert (not self.discrete_env), "DDPG requires continuous environments!"
		algorithm = DDPG( self.gym_env, self.verbose )
		algorithm.loop(ep_step)

	
	def _run_DQN(self, ep_step):
		from BasicRL.DQN import DQN
		assert (self.discrete_env), "DDPG requires discrete environments!"
		algorithm = DQN( self.gym_env, self.verbose )
		algorithm.loop(ep_step)