# Bootstrapping Error Accumulation Reduction (BEAR)
Pytorch implementation of [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/pdf/1906.00949.pdf)

Method is tested on on MuJoCo(mjpro131) continuous control tasks in OpenAI gym, Networks are trained using PyTorch 1.1 and Python 3.6.

| Tested environments: **(`--env_name`)**|
| --------------- |
| HalfCheetah-v1  |
| Hopper-v1       |
| Walker2d-v1     |
| Ant-v1          |
| Humanoid-v1     |

## Training Models

First, train a model used to collect data by running run_sac.py

```
python run_sac.py --env_name HalfCheetah-v1 --buffer_type medio --limit 4000 --seed 1
```

- `HalfCheetah-v1` can be replaced with other environments.
- `--buffer_type` means the mediocre/optimal/random policy according to the paper.
- `--limit` means the mediocre/optimal/random policy stops training when avg_return reaches limit score. 

Next, use the algo to collect 1M dataset by running generate_buffer.py

```
python generate_buffer_sac.py --env_name HalfCheetah-v1 --buffer_type medio --seed 1 --replay_size 1000000
```

Then, we train a prior model D(a|s) of the dataset using MLE by running fit_distribution.py

```
python fit_distribution.py --env_name HalfCheetah-v1 --buffer_type medio --genbuffer_algo SAC --seed 1 --epoch 5000
```

Finally, train BEAR by running run_bear.py

```
CUDA_VISIBLE_DEVICES=0 python run_bear.py --env_name HalfCheetah-v1 --buffer_type medio --genbuffer_algo SAC --seed 1 --init_dual_lambda 100. --dual_steps 10 --cost_epsilon 0.02 --batch_size 1024 --dirac_policy_num 50 --m 5 --n 4 --cuda
```

- `--genbuffer_algo` : which algo to generate buffer.
- `--init_dual_lambda` : init value of dual lambda.
- `--dual_steps` : dual gradient descent times in actor training.
- `--dirac_policy_num` : sample nums of state when updating critic.
- `--m` : sample actor nums when computing mmd.
- `--n` : sample prior nums when computing mmd.

If you want to compare results with BCQ, run as follows:

```
python run_bcq.py --env_name HalfCheetah-v1 --buffer_type medio --seed 1 --genbuffer_algo SAC
```
