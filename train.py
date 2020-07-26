
import cv2
import random
import numpy as np
import argparse
from DRL.evaluator import Evaluator
from utils.util import *
from utils.tensorboard import TensorBoard
import time
import datetime
from options.options import Options

date_and_time = datetime.datetime.now()
run_name = 'painter_' + date_and_time.strftime("%m_%d__%H_%M_%S")
# writer = TensorBoard('train_log_cats/{}'.format(run_name))
writer = TensorBoard('train_log_deleteme/{}'.format(run_name))

if not os.path.exists('model'):
    os.mkdir('model')

def train(agent, env, evaluate):
    train_times = opt.train_times
    env_batch = opt.env_batch
    validate_interval = opt.validate_interval
    max_step = opt.max_step
    debug = opt.debug
    episode_train_times = opt.episode_train_times
    resume = opt.resume
    output = opt.output
    time_stamp = time.time()
    step = episode = episode_steps = 0
    tot_reward = 0.
    observation = None
    noise_factor = opt.noise_factor
    imgs_used_from_file = 0

    while step <= train_times:
        step += 1
        episode_steps += 1
        # reset if it is the start of episode
        if observation is None:
            observation = env.reset()[0]
            agent.reset(observation, noise_factor)

            imgs_used_from_file += agent.batch_size
            if (env.dataset == 'all' and imgs_used_from_file > env.env.train_num):
                env.env.load_new_file()
                print('loading a new file')
                imgs_used_from_file = 0

        action = agent.select_action(observation, noise_factor=noise_factor)
        observation, reward, done, _, mask = env.step(action, episode_steps)
        agent.observe(reward, observation, done, step, mask)
        if (episode_steps >= max_step and max_step):
            if step > opt.warmup:
                # [optional] evaluate
                #if episode > 0 and validate_interval > 0 and episode % validate_interval == 0:
                if validate_interval > 0 and episode % validate_interval == 0:
                    reward, dist = evaluate(env, agent.select_action, debug=debug)
                    if debug: print('Step_{:07d}: mean_reward:{:.3f} mean_dist:{:.3f} var_dist:{:.3f}'\
                        .format(step - 1, np.mean(reward), np.mean(dist), np.var(dist)))
                    writer.add_scalar('validate/mean_reward', np.mean(reward), step)
                    writer.add_scalar('validate/mean_dist', np.mean(dist), step)
                    writer.add_scalar('validate/var_dist', np.var(dist), step)
                    agent.save_model(output)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            tot_Q = 0.
            tot_value_loss = 0.
            if step > opt.warmup:
                # if step < 10000 * max_step:
                #     lr = (9e-4, 3e-3)
                # elif step < 20000 * max_step:
                #     lr = (3e-4, 9e-4)
                # else:
                #     lr = (9e-5, 3e-4)
                # lr = (3e-6, 1e-5)
                lr = (3e-7, 1e-6)
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
    opt = Options().parse()

    opt.output = get_output_folder(opt.output, "Paint")
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    from DRL.ddpg import DDPG
    from DRL.multi import fastenv
    # fenv = fastenv(args.max_step, args.env_batch, writer, args.canvas_color, args.loss_fcn, args.dataset, args.use_multiple_renderers)
    # agent = DDPG(args.batch_size, args.env_batch, args.max_step, \
    #              args.tau, args.discount, args.rmsize, \
    #              writer, args.resume, args.output, args.loss_fcn, args.renderer, args.use_multiple_renderers)
    fenv = fastenv(opt, writer)
    agent = DDPG(opt, writer)
    evaluate = Evaluator(opt, writer)
    print('observation_space', fenv.observation_space, 'action_space', fenv.action_space)

    summary = 'Loss Function - {}\nRenderer - {}\nResuming Model - {}\nbatch_size - {}\nmax_step - {}\nOutput - {}' \
        .format(opt.loss_fcn, opt.renderer, opt.resume, opt.batch_size, opt.max_step, opt.output)
    writer.add_text('summary', summary, 0)
    writer.add_text('Command Line Arguments', str(opt), 0)

    train(agent, fenv, evaluate)
