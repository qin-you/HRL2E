param_id对应.csv中的某套参数，antpush antfall

actor_eval_l, actor_target_l： 前者是当前训练的网络（训练好后用于评估），后者是目标网络

<5, done_h = True <1.5, done_l=True

goal：sg， target：最终目标， 高层输入只有state 没用multigoal做。（是multigoal造成了lunarlander无法到最优点？）

没有使用env reward, 直接使用了负距离作intri reward

done_l不结束当前子任务，阈值关系，可以再走几步小的，或者步伐为0，c=10,也不多

done_h不结束当前子任务，同上，这样一来就不会有阈值圆影响进一步学习了

ant问题没有env_done, 所以存一个high transition只需要判断c耗尽

state_seq 不包含最后一个s，

训练节奏是start_timestep后每新增一个transition训练一次， 第一步goal是用randn得到的，在train,evaluate,video_log中都是

没有舍弃transition，需要较大内存

env创建： 
    utils.py:           get_env(): env = create_maze_env(env_name=env_name)
    creat_maze_env.py:  create_maze_env: gym_env = cls(**gym_mujoco_kwargs), 其中cls = AntMazeEnv
    ant_maze_env.py:    class AntMazeEnv(MazeEnv):    MODEL_CLASS = AntEnv
    maze_env.py:        class MazeEnv: reset():wrapped_env  step():wrapped_env      其中self.wrapped_env = model_cls=self.__class__.MODEL_CLASS
    ant.py:             class AntEnv: reset step  

ensemble 对correction的影响：action_seq不再可以通过直接过一遍actor前向网络得出，而是要ensemble系统和环境真实执行一遍候选goal

correction改到hindsight很容易，且不受ensemble影响。

对correction位置的质疑：加入buffer前做是为了消除噪声影响？没有解决non-stationary问题。

最初版本代码有几个thinkable问题：reward_h,done_h的计算中用了s而非next_s（bug）、 evaluate中一步换一个goal，而不是每c步换一次（应该也对）

cpu tensor 与gpu tensor运算会报错，必须要同一device。 gpu上放的是tensor，即有device= 属性的。 ndarray没有device属性，是因为不放到gpu。
所以，网络模型agent一般放在gpu，replaybuffer可选放在gpu（看用tensor还是ndarray存的）， env肯定在cpu。所以gpu上Actor出来的action，要先
拿到action.detach.cpu才能，env.step(action)

Baseline or Adjustment:

    * hiro: correction position
    * hindsight
    * hrl ensemble: hindsight, correction(maybe slow)

    * smaller buffer: low level:same, high: //3  (non-stationary problem)

    * evaluate with c
    * first goal by network not random

    * bigger noise of high level

args是形参，params是泛指参数，实验优先使用命令行输入的参数设置，没有才到.csv中找

HER以真实到达为目标，永远学不到以障碍内状态为子目标的惩罚，只适合于无障碍开放环境，机械臂，无障碍迷宫。尤其对于dense reward，agent学到的q梯度指向的路径就是直线方向，但却被东西挡住了，这是无法学到的。 0-1奖励

全是负奖励，targetQ突然有正值是因为，actor学习目标正是某状态下在**critic网络**里面找到最大的值，显然它做到了，从某个角落找到了一个正值。
HER在原hiro_ant实验中也有发散的情况

实验阶段总结：实验本想增强低层表现力，提升其在复杂问题中的表现，顺带解决Nonstationary问题。实际上，ensemble本身和Nonstationary无关，单个agentl的表现力已经足够解决问题。现在的问题是：算法收敛时间较长、有个陡减不清楚原理、易陷入次优解探索效率问题采集的h样本可能无意义、目前测试环境为简单环境探索主要靠噪声。
所以希望通过ensemble增加有效探索，加快收敛，增强低层效率，解决Nonstationary或因解决Nonstationary带来的负面问题是高层策略问题。

高层依赖于低层探索，高层低层训练互不相关，理想情况低层能够完美服从高层导向后，低层要做的就是单纯听候高层调遣，探索工作由高层完成。 而训练阶段，低层不能完美服从高层导向，此时它对算法的作用就是勉强作为高层的双脚，也许有意外发现，但这不是应该被期望的。 我们期望的应是一个能完美顺应高层导向的低层策略，越快实现越好。