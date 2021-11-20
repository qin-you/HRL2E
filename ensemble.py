from copy import deepcopy
from xml.etree.ElementTree import QName
import torch
from torch import Tensor
from torch.nn import functional
import numpy as np


class Ensemble_utils:
    def __init__(self):
        self.n_ensemble = 3
        self.cur_agent_ind = torch.randint(0, self.n_ensemble, size=(1,)).item()
        # self.ucb_lamda = 0.1
        # self.epsilon = 0    # we use ucb exploration in replace of e-greedy
        self.mask = torch.tensor([1., 1., 1.])

    # plan B: use 5 TD3 agents to generate a, and use intric reward to score. vote to pick.

    def en_pick_action(self, state, goal, agents, max_action, change, ucb_lamda=1.0):
        # change: whether change cur_agent to generate action
        if change:
            # if torch.rand(1).item() < 1-self.epsilon:
            a_candidate = []
            for agent in agents:
                actor_eval_l = agent['actor_eval_l']
                a = actor_eval_l(state, goal).detach()
                # a_candidate.append(a.clamp(-max_action, max_action).squeeze())
                a_candidate.append(a.squeeze())

            Q_mean = []     # mean
            Q_std = []          # variance
            score_machine = [agent['critic_target_l'] for agent in agents]
            for action in a_candidate:
                score_source = torch.tensor([torch.min(* c(state, goal, action)) for c in score_machine])
                std, mean = torch.std_mean(score_source)
                Q_mean.append(mean)
                Q_std.append(std)

            ucb_list = [m+ucb_lamda*s for m, s in zip(Q_mean, Q_std)]
            ind = ucb_list.index(max(ucb_list))
            ind = torch.randint(0,high=self.n_ensemble, size=(1,)).item()      # if random schedule
            self.cur_agent_ind = ind
            sort_ind = torch.tensor(Q_std).argsort(descending=True)
            _mask = torch.zeros((self.n_ensemble,))
            _mask[sort_ind[:self.n_ensemble]] = 1.      # mask proportion
            self.mask = _mask              
            return a_candidate[ind], self.cur_agent_ind, self.mask
            # else:
            #     ind = torch.randint(0, self.n_ensemble, size=(1,)).item()
            #     actor_eval_l = agents[ind]['actor_eval_l']
            #     self.cur_agent_ind = ind
            #     return actor_eval_l(state, goal).detach(), self.cur_agent_ind
        else:
            actor_eval_l = agents[self.cur_agent_ind]['actor_eval_l']
            # return actor_eval_l(state, goal).detach().clamp(-max_action, max_action), self.cur_agent_ind, self.mask
            return actor_eval_l(state, goal).detach(), self.cur_agent_ind, self.mask

    def en_update(self, experience_buffer, batch_size, total_it, params, agents):
        # sample--update,  n_ensemble times
        target_q_list = []
        critic_loss_list = []
        actor_loss_list = []

        policy_params = params.policy_params
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if params.use_cuda else "cpu"
        total_it[0] += 1
        for ind, agent in enumerate(agents):
            actor_eval = agent['actor_eval_l']
            actor_target = agent['actor_target_l']
            actor_optimizer = agent['actor_optimizer_l']
            critic_eval = agent['critic_eval_l']
            critic_target = agent['critic_target_l']
            critic_optimizer = agent['critic_optimizer_l']
            state, goal, action, reward, next_state, next_goal, done, mask = experience_buffer.sample(batch_size)
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
            # critic_loss = functional.mse_loss(q_eval_1, y) + functional.mse_loss(q_eval_2, y)
            critic_loss = functional.mse_loss(mask[:,ind]*q_eval_1, mask[:,ind]*y) + functional.mse_loss(mask[:,ind]*q_eval_2, mask[:,ind]*y)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            # delayed policy update
            actor_loss = None
            if total_it[0] % policy_params.policy_freq == 0:
                # compute actor loss
                # actor_loss = -critic_eval.q1(state, goal, actor_eval(state, goal)).mean()
                actor_loss = (-critic_eval.q1(state, goal, actor_eval(state, goal)) * mask[:,ind]).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                # soft update: critic q_target
                for param_eval, param_target in zip(critic_eval.parameters(), critic_target.parameters()):
                    param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)
                for param_eval, param_target in zip(actor_eval.parameters(), actor_target.parameters()):
                    param_target.data.copy_(policy_params.tau * param_eval.data + (1 - policy_params.tau) * param_target.data)

            target_q_list.append(y.detach())
            actor_loss_list.append(actor_loss.detach())
            critic_loss_list.append(critic_loss.detach())
        return sum(target_q_list) / len(target_q_list), sum(critic_loss_list) / len(critic_loss_list), sum(actor_loss_list) / len(critic_loss_list)
        

        

# plan A: use 5A1C architecture as TD3, use 5A to generate a, use C to score. vote to pick