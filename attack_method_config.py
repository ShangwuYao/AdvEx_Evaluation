from cleverhans.attacks import *

ATTACK_DICT = {
	'jsma': {'attackmethod': SaliencyMapMethod, 
			'attack_params': {'theta': 1., 'gamma': 0.1,
								'clip_min': 0., 'clip_max': 1.,
								'y_target': None}},
	'mim': {'attackmethod': MomentumIterativeMethod, 
			'attack_params': {'eps': 0.3, 'eps_iter': 0.06, 'nb_iter': 10, 
								'ord': np.inf, 'decay_factor': 1.0,
								'clip_min': 0., 'clip_max': 1.}
			}
}


