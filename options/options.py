import argparse
import os
import torch

"""Authors: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py"""

class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
        parser.add_argument('--discount', default=0.95, type=float, help='discount factor')
        parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
        parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
        parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
        parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
        parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
        parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
        parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
        parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
        parser.add_argument('--train_times', default=10000000, type=int, help='total traintimes')
        parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
        parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
        parser.add_argument('--output', default='./model', type=str, help='Path to directory to output models to')
        parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
        parser.add_argument('--seed', default=1234, type=int, help='random seed')

        parser.add_argument('--loss_fcn', default='cml1', choices=['gan', 'l2', 'l1', 'cm', 'cml1', 'l1_penalized'])
        parser.add_argument('--canvas_color', default='white', choices=['white','black', 'none'])
        parser.add_argument('--renderer', default='renderer.pkl', type=str, help='Filename of renderer used to paint')
        parser.add_argument('--dataset', default='celeba', choices=['celeba','pascal', 'sketchy', 'cats', 'all'])
        parser.add_argument('--use_multiple_renderers', default=False, type=bool, help='Use multiple neural renderers')
        parser.add_argument('--built_in_cm', dest='built_in_cm', action='store_true', help='Build the content masking into the actor model')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Humanoid Painter')
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt
