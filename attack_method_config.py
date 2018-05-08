from cleverhans.attacks import *

ATTACK_DICT = {
	'jsma': {'attackmethod': SaliencyMapMethod, 
			'attack_params': {'theta': 1., 'gamma': 0.1,
								'clip_min': 0., 'clip_max': 1.,
								'y_target': None}
	},
	'mim': {'attackmethod': MomentumIterativeMethod, 
			'attack_params': {'eps': 0.3, 'eps_iter': 0.06, 'nb_iter': 10, 
								'ord': np.inf, 'decay_factor': 1.0,
								'clip_min': 0., 'clip_max': 1.}
	},
	'bim': {'attackmethod': BasicIterativeMethod,
			'attack_params': {'eps':0.3, 'eps_iter':0.05, 'nb_iter':10}
	},
	'lbfgs': {'attackmethod': LBFGS,
			'attack_params': {'batch_size':1,
								 'binary_search_steps':5, 'max_iterations':1000,
								 'initial_const':1e-2, 'clip_min':0, 'clip_max':1}
	},
	'enm': {'attackmethod': ElasticNetMethod,
			'attack_params': {'fista':True, 'beta':1e-3,
								 'decision_rule':'EN', 'batch_size':1, 'confidence':0,
								 'learning_rate':1e-2,
								 'binary_search_steps':9, 'max_iterations':1000,
								 'abort_early':False, 'initial_const':1e-3,
								 'clip_min':0, 'clip_max':1}
	},
	'cw': {'attackmethod': CarliniWagnerL2,
			'attack_params': {'binary_search_steps': 1,
								 'y_target': None,
								 'max_iterations': 10,
								 'learning_rate': 0.1,
								 'batch_size': 100,
								 'initial_const': 10}
	}
}


