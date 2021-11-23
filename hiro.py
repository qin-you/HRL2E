"""
HIRO training process
"""
import os
import datetime
from torch import Tensor
import copy
import torch
from torch.nn import functional
import numpy as np
import wandb
from utils import get_env, get_target_position, log_video_hrl, ParamDict, LoggerTrigger, TimeLogger, print_cmd_hint
from network import ActorLow, ActorHigh, CriticLow, CriticHigh, Gate
from experience_buffer import ExperienceBufferLow, ExperienceBufferHigh, GateBuffer
from ensemble import Ensemble_utils
from copy import deepcopy
from math import exp 
import math
import time


def save_evaluate_utils(step, actor_l, actor_h, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "evalutils-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join(".", "save", "model", file_name)
    print("\n    > saving evaluation utils...")
    torch.save({
        'step': step,
        'actor_l': actor_l.state_dict(),
        'actor_h': actor_h.state_dict(),
    }, file_path)
    print("    > saved evaluation utils to: {}\n".format(file_path))


def save_checkpoint(step, actor_l, critic_l, actor_optimizer_l, critic_optimizer_l, exp_l, actor_h, critic_h, actor_optimizer_h, critic_optimizer_h, exp_h, logger, params, file_path=None, file_name=None):
    if file_name is None:
        time = datetime.datetime.now()
        file_name = "checkpoint-hiro-{}_{}-it({})-[{}].tar".format(params.env_name.lower(), params.prefix, step, time)
    if file_path is None:
        file_path = os.path.join(".", "save", "model", file_name)
    print("\n    > saving training checkpoint...")
    torch.save({
        'step': step,
        'params': params,
        'logger': logger,
        'actor_l': actor_l.state_dict(),
        'critic_l': critic_l.state_dict(),
        'actor_optimizer_l': actor_optimizer_l.state_dict(),
        'critic_optimizer_l': critic_optimizer_l.state_dict(),
        'exp_l': exp_l,
        'actor_h': actor_h.state_dict(),
        'critic_h': critic_h.state_dict(),
        'actor_optimizer_h': actor_optimizer_h.state_dict(),
        'critic_optimizer_h': critic_optimizer_h.state_dict(),
        'exp_h': exp_h
    }, file_path)
    print("    > saved checkpoint to: {}\n".format(file_path))


def load_checkpoint(file_name):
    try:
        # load checkpoint file
        print("\n    > loading training checkpoint...")
        file_path = os.path.join(".", "save", "model", file_name)
        checkpoint = torch.load(file_path)
        print("\n    > checkpoint file loaded! parsing data...")
        params = checkpoint['params']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
        # load utils
        policy_params = params.policy_params
        state_dim = params.state_dim
        goal_dim = params.goal_dim
        action_dim = params.action_dim
        max_action = policy_params.max_action
        max_goal = policy_params.max_goal
        # initialize rl components
        actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, max_action).to(device)
        actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
        critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
        critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
        actor_eval_h = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
        actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
        critic_eval_h = CriticHigh(state_dim, goal_dim).to(device)
        critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
        # unpack checkpoint object
        step = checkpoint['step'] + 1
        logger = checkpoint['logger']
        #
        actor_eval_l.load_state_dict(checkpoint['actor_l'])
        critic_eval_l.load_state_dict(checkpoint['critic_l'])
        actor_optimizer_l.load_state_dict((checkpoint['actor_optimizer_l']))
        critic_optimizer_l.load_state_dict(checkpoint['critic_optimizer_l'])
        experience_buffer_l = checkpoint['exp_l']
        #
        actor_eval_h.load_state_dict(checkpoint['actor_h'])
        critic_eval_h.load_state_dict(checkpoint['critic_h'])
        actor_optimizer_h.load_state_dict((checkpoint['actor_optimizer_h']))
        critic_optimizer_h.load_state_dict(checkpoint['critic_optimizer_h'])
        experience_buffer_h = checkpoint['exp_h']
        #
        actor_target_l = copy.deepcopy(actor_eval_l).to(device)
        critic_target_l = copy.deepcopy(critic_eval_l).to(device)
        actor_target_h = copy.deepcopy(actor_eval_h).to(device)
        critic_target_h = copy.deepcopy(critic_eval_h).to(device)
        #
        actor_eval_l.train(), actor_target_l.train(), critic_eval_l.train(), critic_target_l.train()
        actor_eval_h.train(), actor_target_h.train(), critic_eval_h.train(), critic_target_h.train()
        print("    > checkpoint resume success!")
    except Exception as e:
        print(e)
    return [step, params, device, logger,
            actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h]


def initialize_params(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    evaluation_interval = params.evaluation_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    video_log_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    state_print_trigger = LoggerTrigger(start_ind=policy_params.start_timestep)
    checkpoint_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    evalutil_logger = LoggerTrigger(start_ind=policy_params.start_timestep, first_log=False)
    time_logger = TimeLogger()
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger]


def initialize_params_checkpoint(params, device):
    policy_params = params.policy_params
    env_name = params.env_name
    max_goal = Tensor(policy_params.max_goal).to(device)
    action_dim = params.action_dim
    goal_dim = params.goal_dim
    max_action = policy_params.max_action
    expl_noise_std_l = policy_params.expl_noise_std_l
    expl_noise_std_h = policy_params.expl_noise_std_h
    c = policy_params.c
    episode_len = policy_params.episode_len
    max_timestep = policy_params.max_timestep
    start_timestep = policy_params.start_timestep
    batch_size = policy_params.batch_size
    log_interval = params.log_interval
    checkpoint_interval = params.checkpoint_interval
    save_video = params.save_video
    video_interval = params.video_interval
    env = get_env(params.env_name)
    return [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
            c, episode_len, max_timestep, start_timestep, batch_size,
            log_interval, checkpoint_interval, save_video, video_interval, env]


def record_logger(args, option, step):
    if option == "inter_loss":
        target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h, gate_loss, gate_score_mean, gate_score_std = args[:]
        if target_q_l is not None: wandb.log({'target_q low': torch.mean(target_q_l).squeeze()}, step=step)
        if critic_loss_l is not None: wandb.log({'critic_loss low': torch.mean(critic_loss_l).squeeze()}, step=step)
        if actor_loss_l is not None: wandb.log({'actor_loss low': torch.mean(actor_loss_l).squeeze()}, step=step)
        if target_q_h is not None: wandb.log({'target_q high': torch.mean(target_q_h).squeeze()}, step=step)
        if critic_loss_h is not None: wandb.log({'critic_loss high': torch.mean(critic_loss_h).squeeze()}, step=step)
        if actor_loss_h is not None: wandb.log({'actor_loss high': torch.mean(actor_loss_h).squeeze()}, step=step)
        if gate_loss is not None: wandb.log({'gate_loss': torch.mean(gate_loss).squeeze()}, step=step)
        if gate_score_mean is not None: wandb.log({'gate_score_mean': torch.mean(gate_score_mean).squeeze()}, step=step)
        if gate_score_std is not None: wandb.log({'gate_score_std': torch.mean(gate_score_std)}, step=step)
    elif option == "reward":
        episode_reward_l, episode_reward_h = args[:]
        wandb.log({'episode reward low': episode_reward_l}, step=step)
        wandb.log({'episode reward high': episode_reward_h}, step=step)
    elif option == "success_rate":
        success_rate = args[0]
        wandb.log({'success rate': success_rate}, step=step)


def create_rl_components(params, device):
    # function local utils
    policy_params = params.policy_params
    state_dim, goal_dim, action_dim = params.state_dim, params.goal_dim, params.action_dim
    max_goal = Tensor(policy_params.max_goal)
    # low-level
    step, episode_num_h = 0, 0
    actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, policy_params.max_action).to(device)
    actor_target_l = copy.deepcopy(actor_eval_l).to(device)
    actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
    critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
    critic_target_l = copy.deepcopy(critic_eval_l).to(device)
    critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
    experience_buffer_l = ExperienceBufferLow(int(policy_params.max_timestep / 10), state_dim, goal_dim, action_dim, params.use_cuda)
    # high-level
    actor_eval_h = ActorHigh(state_dim, goal_dim, max_goal, device).to(device)
    actor_target_h = copy.deepcopy(actor_eval_h).to(device)
    actor_optimizer_h = torch.optim.Adam(actor_eval_h.parameters(), lr=policy_params.actor_lr)
    critic_eval_h = CriticHigh(state_dim, goal_dim).to(device)
    critic_target_h = copy.deepcopy(critic_eval_h).to(device)
    critic_optimizer_h = torch.optim.Adam(critic_eval_h.parameters(), lr=policy_params.critic_lr)
    experience_buffer_h = ExperienceBufferHigh(int(policy_params.max_timestep / policy_params.c / 3) + 1, state_dim, goal_dim, params.use_cuda, policy_params.c, action_dim)
    
    return [step, episode_num_h,
            actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
            actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h]

def create_en_agents(params, device, n_en):
    # create ensemble low agent: A C 
    en_agents = []
    policy_params = params.policy_params
    state_dim, goal_dim, action_dim = params.state_dim, params.goal_dim, params.action_dim
    gates = []
    for _ in range(n_en):
        gate_buffer = GateBuffer(int(params.policy_params.max_timestep / params.policy_params.c / 18) + 1, params.goal_dim, params.goal_dim, 1, params.use_cuda)
        gate_net = Gate(goal_dim+goal_dim, n_en).to(device)
        gate_optimizer = torch.optim.Adam(gate_net.parameters(), lr=policy_params.actor_lr) 
        temp = {'gate_buffer': gate_buffer, 'gate_net': gate_net, 'gate_optimizer':gate_optimizer}
        gates.append(temp)
    for i in range(n_en):
        actor_eval_l = ActorLow(state_dim, goal_dim, action_dim, policy_params.max_action).to(device)
        actor_target_l = copy.deepcopy(actor_eval_l).to(device)
        actor_optimizer_l = torch.optim.Adam(actor_eval_l.parameters(), lr=policy_params.actor_lr)
        critic_eval_l = CriticLow(state_dim, goal_dim, action_dim).to(device)
        critic_target_l = copy.deepcopy(critic_eval_l).to(device)
        critic_optimizer_l = torch.optim.Adam(critic_eval_l.parameters(), lr=policy_params.critic_lr)
        temp = {'actor_eval_l':actor_eval_l, 'actor_target_l':actor_target_l, 'actor_optimizer_l':actor_optimizer_l, 
                'critic_eval_l':critic_eval_l, 'critic_target_l':critic_target_l, 'critic_optimizer_l':critic_optimizer_l}
        en_agents.append(temp)
    return en_agents, gates




def h_function(state, goal, next_state, goal_dim):
    # return next goal
    return state[:goal_dim] + goal - next_state[:goal_dim]


def intrinsic_reward(state, goal, next_state):
    # low-level dense reward (L2 norm), provided by high-level policy
    return -torch.pow(sum(torch.pow(state + goal - next_state, 2)), 1 / 2)


def intrinsic_reward_simple(state, goal, next_state, goal_dim):
    # low-level dense reward (L2 norm), provided by high-level policy
    return -torch.pow(sum(torch.pow(state[:goal_dim] + goal - next_state[:goal_dim], 2)), 1 / 2)

def heuristic_intrinsic_reward(state, goal, next_state, goal_dim, goal0):
    d = torch.norm(state[:goal_dim] + goal - next_state[:goal_dim])
    a = next_state[:goal_dim] - state[:goal_dim]
    b = goal
    cos = (a*b).sum() / (torch.norm(a) * torch.norm(b))
    # cos = (a*b).sum() / (torch.pow(torch.pow(a,2).sum(), 1/2) * torch.pow(torch.pow(b,2).sum(), 1/2))
    return -d * exp(-cos)

def gate_score_cal(start_state, end_state, goal, goal_dim):
    d = torch.norm(end_state[:goal_dim] - start_state[:goal_dim])
    a = end_state[:goal_dim] - start_state[:goal_dim]
    b = goal
    cos = (a*b).sum() / (torch.norm(a) * torch.norm(b))
    return d*cos

def dense_reward(state, goal_dim, target=Tensor([0, 19, 0.5])):
    device = state.device
    target = target.to(device)
    l2_norm = torch.pow(sum(torch.pow(state[:goal_dim] - target, 2)), 1 / 2)
    return -l2_norm


def done_judge_low(goal):
    # define low-level success: same as high-level success (L2 norm < 5, paper B.2.2)
    l2_norm = torch.pow(sum(torch.pow(goal, 2)), 1 / 2)
    # done = (l2_norm <= 5.)
    done = (l2_norm <= 1.5)
    return Tensor([done])


def success_judge(state, goal_dim, target=Tensor([0, 19, 0.5])):
    location = state[:goal_dim]
    l2_norm = torch.pow(sum(torch.pow(location - target, 2)), 1 / 2)
    done = (l2_norm <= 5.)
    return Tensor([done])


def off_policy_correction(actor, action_sequence, state_sequence, goal_dim, goal, end_state, max_goal, device):
    # Hindsight
    # mean = (end_state - state_sequence[0])[:goal_dim].cpu()
    # return mean.to(device), True

    # off-policy correction
    # action_sequence = torch.stack(action_sequence).to(device)
    # state_sequence = torch.stack(state_sequence).to(device)
    max_goal = max_goal.cpu()
    # prepare candidates
    mean = (end_state - state_sequence[0])[:goal_dim].cpu()
    std = 0.5 * max_goal
    candidates = [torch.min(torch.max(Tensor(np.random.normal(loc=mean, scale=std, size=goal_dim).astype(np.float32)), -max_goal), max_goal) for _ in range(8)]
    candidates.append(mean)
    candidates.append(goal.cpu())
    # select maximal
    candidates = torch.stack(candidates).to(device)
    surr_prob = [-functional.mse_loss(action_sequence, actor(state_sequence, state_sequence[0][:goal_dim] + candidate - state_sequence[:, :goal_dim])) for candidate in candidates]
    index = int(np.argmax(surr_prob))
    updated = (index != 9)
    goal_hat = candidates[index]
    return goal_hat.cpu(), updated

def correction_before_train(actor, action_arr, state_arr, goal_dim, goal_arr, end_states, max_goal, device, batch_size):
    # batchsize * dim or batchsize*c*dim
    goal_hat_arr = torch.zeros(batch_size, goal_dim)
    for i in range(batch_size):
        goal, _ = off_policy_correction(actor, action_arr[i], state_arr[0], goal_dim, goal_arr[i], end_states[i], max_goal, device)
        goal_hat_arr[i] = goal
    return goal_hat_arr


def step_update_l(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, params):
    # initialize
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    total_it[0] += 1
    # sample mini-batch transitions
    state, goal, action, reward, next_state, next_goal, done = experience_buffer.sample(batch_size)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.action_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_action = (actor_target(next_state, next_goal) + policy_noise).clamp(-policy_params.max_action, policy_params.max_action)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(next_state, next_goal, next_action)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_l * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state, goal, action)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy update
    actor_loss = None
    if total_it[0] % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state, goal, actor_eval(state, goal)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss


def step_update_h(experience_buffer, batch_size, total_it, actor_eval, actor_target, critic_eval, critic_target, critic_optimizer, actor_optimizer, actor_l, params):
    policy_params = params.policy_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    max_goal = Tensor(policy_params.max_goal).to(device)
    goal_dim = params.goal_dim
    # sample mini-batch transitions
    state_start, goal_arr, reward, state_end, done, state_arr, action_arr = experience_buffer.sample(batch_size)
    # correction
    goal = correction_before_train(actor_l, action_arr, state_arr, goal_dim, goal_arr, state_end, max_goal, device, batch_size).to(device)
    with torch.no_grad():
        # select action according to policy and add clipped noise
        policy_noise = Tensor(np.random.normal(loc=0, scale=policy_params.policy_noise_std, size=params.goal_dim).astype(np.float32) * policy_params.policy_noise_scale) \
            .clamp(-policy_params.policy_noise_clip, policy_params.policy_noise_clip).to(device)
        next_goal = torch.min(torch.max(actor_target(state_end) + policy_noise, -max_goal), max_goal)
        # clipped double Q-learning
        q_target_1, q_target_2 = critic_target(state_end, next_goal)
        q_target = torch.min(q_target_1, q_target_2)
        y = policy_params.reward_scal_h * reward + (1 - done) * policy_params.discount * q_target
    # update critic q_evaluate
    q_eval_1, q_eval_2 = critic_eval(state_start, goal)
    critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    # delayed policy updates
    actor_loss = None
    if int(total_it[0] / policy_params.c) % policy_params.policy_freq == 0:
        # compute actor loss
        actor_loss = -critic_eval.q1(state_start, actor_eval(state_start)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        # soft update: critic q_target
        for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
            param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
        actor_loss = actor_loss.detach()
    return y.detach(), critic_loss.detach(), actor_loss

def step_update_gate(gates, batch_size, total_it, cur_ind, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    loss, s_std, s_mean = [], [], []
    for gate in gates:
        X, y = gate['gate_buffer'].sample(batch_size)
        scores_prediction = gate['gate_net'](X)
        loss = functional.mse_loss(scores_prediction, y)
        gate['gate_optimizer'].zero_grad()
        loss.backward()
        gate['gate_optimizer'].step()
        score_std, score_mean = torch.std_mean(scores_prediction)
        loss.append(loss.detach().item())
        s_std.append(score_std.item())
        s_mean.append(score_mean.item())
    return sum(loss)/len(loss), sum(s_std)/len(s_std), sum(s_mean)/len(s_mean)
    

def evaluate(agents_l, en_utils, actor_h, params, target_pos, gates, device):
    labels = [str(i) for i in range(len(agents_l))]
    values = [0 for i in range(len(agents_l))]
    policy_params = params.policy_params
    print("\n    > evaluating policies...")
    success_number = 0
    env = get_env(params.env_name)
    goal_dim = params.goal_dim
    # en_utils.epsilon = 0
    for i in range(4):
        env.seed(policy_params.seed + i)
        for j in range(5):
            t = 0
            episode_len = policy_params.episode_len
            state, done = Tensor(env.reset()).to(device), False
            goal = Tensor(torch.randn(goal_dim)).to(device)
            # goal = actor_h(state)
            while not done and t < episode_len:
                t += 1
                # action = actor_l(obs, goal).to(device)              
                action = en_utils.en_pick_action(state, goal, agents_l, params.policy_params.max_action, change=True, steps=None, goal_dim=goal_dim, epsilon=0, ucb_lamda=0., gates=gates, option='gate', gate_pretrain_steps=math.inf)[0]
                values[en_utils.cur_agent_ind] += 1
                next_state, _, _, _ = env.step(action.detach().cpu())
                next_state = Tensor(next_state).to(device)
                done = success_judge(next_state, goal_dim, target_pos)
                goal = actor_h(next_state)
                state = next_state
            if done:
                success_number += 1
        print("        > evaluated {} episodes".format(i * 5 + j + 1))
    data = [[label, val] for (label, val) in zip(labels, values)]
    table = wandb.Table(data=data, columns = ["label", "value"])
    wandb.log({"agent use" : wandb.plot.bar(table, "label", "value", title="agent use count")})
    success_rate = success_number / 20
    print("    > finished evaluation, success rate: {}\n".format(success_rate))
    return success_rate


def train(params):
    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
    en_utils = Ensemble_utils()
    if params.checkpoint is None:
        # > rl components
        [step, episode_num_h,
         actor_eval_l, actor_target_l, actor_optimizer_l, critic_eval_l, critic_target_l, critic_optimizer_l, experience_buffer_l,
         actor_eval_h, actor_target_h, actor_optimizer_h, critic_eval_h, critic_target_h, critic_optimizer_h, experience_buffer_h] = create_rl_components(params, device)
        en_agents, gates = create_en_agents(params, device, en_utils.n_ensemble)
        
        # en_agents = [{'actor_eval_l':actor_eval_l, 'actor_target_l':actor_target_l, 'actor_optimizer_l':actor_optimizer_l,
        #                 'critic_eval_l':critic_eval_l, 'critic_target_l':critic_target_l, 'critic_optimizer_l':critic_optimizer_l}]
        # gates = [{'gate_buffer': gate_buffer, 'gate_net': gate_net, 'gate_optimizer':gate_optimizer}]
        # > running utils
        [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
         c, episode_len, max_timestep, start_timestep, batch_size,
         log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env, video_log_trigger, state_print_trigger, checkpoint_logger, evalutil_logger, time_logger] = initialize_params(params, device)
    else:
        # > rl components
        pass
        # prefix = params.prefix
        # [step, params, device, [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h],
        #  actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,
        #  actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h] = load_checkpoint(params.checkpoint)
        # # > running utils
        # [policy_params, env_name, max_goal, action_dim, goal_dim, max_action, expl_noise_std_l, expl_noise_std_h,
        #  c, episode_len, max_timestep, start_timestep, batch_size,
        #  log_interval, checkpoint_interval, evaluation_interval, save_video, video_interval, env] = initialize_params_checkpoint(params, device)
        # params.prefix = prefix
    target_q_h, critic_loss_h, actor_loss_h, gate_loss, gate_score_mean, gate_score_std = None, None, None, None, None, None
    target_pos = get_target_position(env_name).to(device)
    # 1.2 set seeds
    env.seed(policy_params.seed)
    torch.manual_seed(policy_params.seed)
    np.random.seed(policy_params.seed)

    # 2. Training Algorithm (TD3)
    # 2.1 initialize
    print_cmd_hint(params=params, location='start_train')
    time_logger.time_spent()
    total_it = [0]
    success_rate, episode_reward_l, episode_reward_h, episode_reward, episode_num_l, episode_timestep_l, episode_timestep_h = 0, 0, 0, 0, 0, 1, 1
    state = Tensor(env.reset()).to(device)
    goal = Tensor(torch.randn(goal_dim)).to(device)
    state_sequence, goal_sequence, action_sequence, intri_reward_sequence, reward_h_sequence = [], [], [], [], []
    # 2.2 training loop
    for t in range(step, max_timestep):
        # 2.2.1 sample action
        if t < start_timestep:
            action = env.action_space.sample()
            mask = torch.tensor([1.,1.,1.], device=device)
        else:
            expl_noise_action = np.random.normal(loc=0, scale=expl_noise_std_l, size=action_dim).astype(np.float32)
            # action = (actor_eval_l(state, goal).detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
            a_tmp, mask = en_utils.en_pick_action(state, goal, en_agents, max_action, (t+1)%c==1, t, goal_dim, epsilon=0.9, ucb_lamda=10., gates=gates, option='gate',gate_pretrain_steps=t-start_timestep)     # episode_timestep_h==1      (t+1)%c==1
            action = (a_tmp.detach().cpu() + expl_noise_action).clamp(-max_action, max_action).squeeze()
        # 2.2.2 interact environment
        next_state, _, _, info = env.step(action)
        next_state = Tensor(next_state).to(device)
        # 2.2.3 compute step arguments
        reward_h = dense_reward(next_state, goal_dim, target=target_pos)     
        done_h = success_judge(next_state, goal_dim, target_pos)
        # reward_h = 1 if done_h else 0           # 0-1 reward for experiment1
        action, reward_h, done_h = Tensor(action), Tensor([reward_h]), Tensor([done_h])
        goal_sequence.append(goal)
        intri_reward = intrinsic_reward_simple(state, goal, next_state, goal_dim)
        # intri_reward = heuristic_intrinsic_reward(state, goal, next_state, goal_dim, goal0=goal_sequence[0])     
        next_goal = h_function(state, goal, next_state, goal_dim)
        done_l = done_judge_low(goal)
        # 2.2.4 collect low-level experience
        experience_buffer_l.add(state, goal, action, intri_reward, next_state, next_goal, done_l, mask)
        # 2.2.5 record segment arguments
        state_sequence.append(state)
        action_sequence.append(action)
        intri_reward_sequence.append(intri_reward)
        reward_h_sequence.append(reward_h)
        # 2.2.6 update low-level segment reward
        episode_reward_l += intri_reward
        episode_reward_h += reward_h
        episode_reward += reward_h
        if (t + 1) % c == 0 and t > 0:
            # 2.2.7 sample goal
            if t < start_timestep:
                next_goal = (torch.randn_like(goal) * max_goal)
                next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
            else:
                expl_noise_goal = np.random.normal(loc=0, scale=expl_noise_std_h, size=goal_dim).astype(np.float32)
                next_goal = (actor_eval_h(next_state.to(device)).detach().cpu() + expl_noise_goal).squeeze().to(device)
                next_goal = torch.min(torch.max(next_goal, -max_goal), max_goal)
            # 2.2.8 collect high-level experience and gate experience
            # goal_hat, updated = off_policy_correction(en_agents[0]['actor_target_l'], action_sequence, state_sequence, goal_dim, goal_sequence[0], next_state, max_goal, device)
            state_arr, action_arr = torch.stack(state_sequence), torch.stack(action_sequence)
            experience_buffer_h.add(state_sequence[0], goal_sequence[0], episode_reward_h, next_state, done_h, state_arr, action_arr)
            if t >= start_timestep:
                gate_label = intri_reward
                gates[en_utils.cur_agent_ind]['gate_buffer'].add(state_sequence[0][:goal_dim], goal_sequence[0], label=gate_label)
            # if state_print_trigger.good2log(t, 500): print_cmd_hint(params=[state_sequence, goal_sequence, action_sequence, intri_reward_sequence, updated, goal_hat, reward_h_sequence], location='training_state')
            # 2.2.9 reset segment arguments & log (reward)
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []   
            print(f"    > Segment: Total T: {t + 1} Episode_L Num: {episode_num_l + 1} Episode_L T: {episode_timestep_l} Reward_L: {float(episode_reward_l):.3f} Reward_H: {float(episode_reward_h):.3f}")
            if t >= start_timestep: record_logger(args=[episode_reward_l, episode_reward_h], option='reward', step=t-start_timestep)
            episode_reward_l, episode_timestep_l = 0, 0
            episode_reward_h = 0
            episode_num_l += 1
        # 2.2.10 update observations
        state = next_state
        goal = next_goal

        # 2.2.11 update networks
        if t >= start_timestep:                                              
            # target_q_l, critic_loss_l, actor_loss_l = \
            #     step_update_l(experience_buffer_l, batch_size, total_it, actor_eval_l, actor_target_l, critic_eval_l, critic_target_l, critic_optimizer_l, actor_optimizer_l, params)
            target_q_l, critic_loss_l, actor_loss_l = en_utils.en_update(experience_buffer_l, batch_size, total_it, params, en_agents) 
        if t >= start_timestep and (t + 1) % c == 0:
            target_q_h, critic_loss_h, actor_loss_h = \
                step_update_h(experience_buffer_h, batch_size, total_it, actor_eval_h, actor_target_h, critic_eval_h, critic_target_h, critic_optimizer_h, actor_optimizer_h, en_agents[0]['actor_target_l'], params)
        if t >= start_timestep + en_utils.gate_pretrain_threshold and (t + 1) % (3*c) == 0:
            gate_loss, gate_score_std, gate_score_mean = step_update_gate(gates, batch_size, total_it, en_utils.cur_agent_ind, params)
        # 2.2.12 log training curve (inter_loss)
        if t >= start_timestep and t % log_interval == 0:
            record_logger(args=[target_q_l, critic_loss_l, actor_loss_l, target_q_h, critic_loss_h, actor_loss_h, gate_loss, gate_score_mean, gate_score_std], option='inter_loss', step=t-start_timestep)
            record_logger([success_rate], 'success_rate', step=t - start_timestep)
        # 2.2.13 start new episode
        if episode_timestep_h >= episode_len:
            # > update loggers
            if t > start_timestep: episode_num_h += 1
            else: episode_num_h = 0
            print(f"    >>> Episode: Total T: {t + 1} Episode_H Num: {episode_num_h+1} Episode_H T: {episode_timestep_h} Reward_Episode: {float(episode_reward):.3f}\n")
            # > clear loggers
            episode_reward = 0
            state_sequence, action_sequence, intri_reward_sequence, goal_sequence, reward_h_sequence = [], [], [], [], []
            episode_reward_l, episode_timestep_l, episode_num_l = 0, 0, 0
            state, done_h = Tensor(env.reset()).to(device), Tensor([False])
            episode_reward_h, episode_timestep_h = 0, 0
        # 2.2.14 update training loop arguments
        episode_timestep_l += 1
        episode_timestep_h += 1
        # 2.2.15 save videos & checkpoints
        if save_video and video_log_trigger.good2log(t, video_interval):
            log_video_hrl(env_name, en_agents, deepcopy(en_utils), actor_target_h, gates, params)     
            time_logger.sps(t)
            time_logger.time_spent()
            print("")
        if False and checkpoint_logger.good2log(t, checkpoint_interval):    # don't save medial checkpoint(900M / model)
            pass
            # logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
            # save_checkpoint(t, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,     # TODO
            #                 actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
            #                 logger, params)
        if t > start_timestep and evalutil_logger.good2log(t, evaluation_interval):
            success_rate = evaluate(en_agents, deepcopy(en_utils), actor_target_h, params, target_pos, gates, device)                 
    # 2.3 final log (episode videos)
    logger = [time_logger, state_print_trigger, video_log_trigger, checkpoint_logger, evalutil_logger, episode_num_h]
    # save_checkpoint(max_timestep, actor_target_l, critic_target_l, actor_optimizer_l, critic_optimizer_l, experience_buffer_l,      
    #                 actor_target_h, critic_target_h, actor_optimizer_h, critic_optimizer_h, experience_buffer_h,
    #                 logger, params)
    for i in range(3):
        log_video_hrl(env_name, en_agents, actor_target_h, gates, params)         
    print_cmd_hint(params=params, location='end_train')


if __name__ == "__main__":
    env_name = "AntMaze"
    env = get_env(env_name)
    state_dim = env.observation_space.shape[0]
    goal_dim = 3
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_goal = [10., 10., .5]
    policy_params = ParamDict(
        seed=54321,
        c=10,
        policy_noise_scale=0.2,
        policy_noise_std=1.,
        expl_noise_std_l=1.,
        expl_noise_std_h=1.,
        policy_noise_clip=0.5,
        max_action=max_action,
        max_goal=max_goal,
        discount=0.99,
        policy_freq=1,
        tau=5e-3,
        actor_lr=1e-4,
        critic_lr=1e-3,
        reward_scal_l=1.,
        reward_scal_h=.1,
        episode_len=1000,
        max_timestep=int(3e6),
        start_timestep=int(300),
        batch_size=100
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=env_name,
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=3,
        video_interval=int(1e4),
        log_interval=5,
        checkpoint_interval=int(1e5),
        evaluation_interval=int(1e4),
        prefix="test_simple_origGoal_fixedIntriR_posER",
        save_video=True,
        use_cuda=True,
        # checkpoint="hiro-antpush_test_simple_origGoal_fixedIntriR_posER-it(2000000)-[2020-07-02 20:35:25.673267].tar"
        checkpoint=None
    )

    # wandb.init(project="hrl_ensemble")
    train(params=params)
