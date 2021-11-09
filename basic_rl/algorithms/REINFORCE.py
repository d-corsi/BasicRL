from basic_rl.abstract_class.continuous_policy_gradient import ContinuousPolicyGradient
from basic_rl.abstract_class.discrete_policy_gradient import DiscretePolicyGradient
import numpy as np

class DiscreteREINFORCE( DiscretePolicyGradient ):

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)


class ContinuousREINFORCE( ContinuousPolicyGradient ):

	def __init__(self, env, **kwargs):
		super().__init__(env)

		for key, value in kwargs.items():
			if hasattr(self, key) and value is not None: 
				setattr(self, key, value)
				