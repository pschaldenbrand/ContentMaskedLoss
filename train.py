
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time
import datetime

date_and_time = datetime.datetime.now()
run_name = 'painter_' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('train_log_multiple_renderers/{}'.format(run_name))

if not os.path.exists('model'):
    os.mkdir('model')

def train(agent, env, evaluate):
    train_times = args.train_times
    env_batch = args.env_batch
    validate_interval = args.validate_interval
    max_step = args.max_step
    debug = args.debug
    episode_train_times = args.episode_train_times
    resume = args.resume
    output = args.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = args.noise_factor
    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()
            agent.reset(observation, noise_factor)    
        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _ = env.step(action, episode_steps)
        agent.observe(reward, observation, done, step)
        if (episode_steps >= max_step and max_step):
            if step > args.warmup:
                # [optional] evaluate
                if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: print('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'.format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > args.warmup:
                lr = (3e-6, 1e-5)
                # if step < 10000 * max_step:
                #     lr = (3e-4, 1e-3)
                # elif step < 20000 * max_step:
                #     lr = (1e-4, 3e-4)
                # else:
                #     lr = (3e-5, 1e-4)
                for i in range(episode_train_times):
                    Q, value_loss = agent.update_policy(lr, episode_steps)
                    tot_Q += Q.data.cpu().numpy()
                    tot_value_loss += value_loss.data.cpu().numpy()
                writer.add_scalar('train/critic_lr', lr[0], step)
                writer.add_scalar('train/actor_lr', lr[1], step)
                writer.add_scalar('train/Q', tot_Q / episode_train_times, step)
                writer.add_scalar('train/critic_loss', tot_value_loss / episode_train_times, step)
            if debug: print('#{}: steps:{} interval_time:{:.2f} train_time:{:.2f}' \
                .format(episode, step, train_time_interval, time.time()-time_stamp)) 
            time_stamp = time.time()
            # reset
            observation = None
            episode_steps = 0
            episode += 1
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning to Paint')

    # hyper-parameter
    parser.add_argument('--warmup', default=400, type=int, help='timestep without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.95**5, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=96, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=800, type=int, help='replay memory size')
    parser.add_argument('--env_batch', default=96, type=int, help='concurrent environment number')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
    parser.add_argument('--noise_factor', default=0, type=float, help='noise level for parameter space noise')
    parser.add_argument('--validate_interval', default=50, type=int, help='how many episodes to perform a validation')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validation')
    parser.add_argument('--train_times', default=2000000, type=int, help='total traintimes')
    parser.add_argument('--episode_train_times', default=10, type=int, help='train times for each episode')    
    parser.add_argument('--resume', default=None, type=str, help='Resuming model path for testing')
    parser.add_argument('--output', default='./model', type=str, help='Path to directory to output models to')
    parser.add_argument('--debug', dest='debug', action='store_true', help='print some info')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')

    parser.add_argument('--loss_fcn', default='cml1', choices=['gan', 'l2', 'l1', 'cm', 'cml1'])
    parser.add_argument('--canvas_color', default='white', choices=['white','black'])
    parser.add_argument('--renderer', default='renderer.pkl', type=str, help='Filename of renderer used to paint')
    parser.add_argument('--dataset', default='celeba', choices=['celeba','pascal', 'sketchy'])
    parser.add_argument('--use_multiple_renderers', default=False, type=bool, help='Use multiple neural renderers')

    args = parser.parse_args()    
    args.output = get_output_folder(args.output, "Paint")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    fenv = fastenv(args.max_step, args.env_batch, writer, args.canvas_color, args.loss_fcn, args.dataset, args.use_multiple_renderers)
    agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
                 args.tau, args.discount, args.rmsize, \
                 writer, args.resume, args.output, args.loss_fcn, args.renderer, args.use_multiple_renderers)
    evaluate = Evaluator(args, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)

    summary = 'Loss Function - {}\nRenderer - {}\nResuming Model - {}\nbatch_size - {}\nmax_step - {}\nOutput - {}' \
        .format(args.loss_fcn, args.renderer, args.resume, args.batch_size, args.max_step, args.output)
    writer.add_text('summary', summary, 0)
    writer.add_text('Command Line Arguments', str(args), 0)

    train(agent, fenv, evaluate)
