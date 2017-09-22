import json
from collections import namedtuple


def parse_arguments(in_hp={}, in_evaluation={}, in_run={}):
    """Load configuration from JSON files in parameters directory. Optionally supply parameters to
    override/supplement the parameters loaded from the files in the parameters directory.

    Args:
	in_hp: A dictionary of hyperparameter settings.
	in_evaluation: A dictionary of evaluation parameters.
	in_run: A dictionary of run parameters`
    
    Returns:
        a five-tuple of dictionaries:
	    hp: A hyperparameter dictionary.
	    evaluation: A dictionary of evaluation parameters.
	    run: A dictionary of run parameters.
	    env: A dictionary of environment parameters.
	    design: A dictionary of design parameters (e.g. network architecture)
    """

    with open('parameters/hyperparams.json') as json_file:
        hp = json.load(json_file)
    with open('parameters/evaluation.json') as json_file:
        evaluation = json.load(json_file)
    with open('parameters/run.json') as json_file:
        run = json.load(json_file)
    with open('parameters/environment.json') as json_file:
        env = json.load(json_file)
    with open('parameters/design.json') as json_file:
        design = json.load(json_file)                

    for name,value in in_hp.iteritems():
        hp[name] = value
    for name,value in in_evaluation.iteritems():
        evaluation[name] = value
    for name,value in in_run.iteritems():
        run[name] = value
    
    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
