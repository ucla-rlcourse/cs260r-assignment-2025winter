# Mini Project

*CS260R: Reinforcement Learning. Department of Computer Science at University of California, Los Angeles.*

-----

## **Overview**

The goal of this assignment is to apply your knowledge of reinforcement learning (RL) to train a competitive RL agent
that performs well in the **MetaDrive Safety Environment**. Before that, you also need to conduct the generalization
experiment with your selected algorithm.

You are free to choose **any RL algorithm or codebase** to train your agent.

### **Generalization Experiment**

- Train your agent in a set of **training environments** (with 1, 3, or 10 unique maps).
- Evaluate the performance of your trained agents in the **validation environment** as well as their original training
  environments.
- Plot the evaluation results of your agents in the training and validation environments.
    - The X-axis represents the number of training maps and the Y-axis represents the evaluation metric. You can try
      different merics, such as `episode_reward` or `route_completion`.
    - We expect two lines in the figure, showing the final training performance and the validation performance.
    - You can refer to the Figure 8 in  [MetaDrive paper](https://arxiv.org/pdf/2109.12674.pdf) and Figure 2
      in [ProcGen paper](http://proceedings.mlr.press/v97/cobbe19a/cobbe19a.pdf).
- Discuss the generalization experiment in your report.
- We only need the curves, do not submit training logs, agents or checkpoints related to this generalization experiment
  in your agent zip file. But you need to upload the training code to the code zip file.

You can define the environments used in this experiment by changing `num_scenarios` in the environment config to be 1,
3, or 10.

For example, you can:

```python
# This function is defined in `env.py`:
def get_training_env(extra_config=None):
    config = copy.deepcopy(TRAINING_CONFIG)
    if extra_config:
        config.update(extra_config)
    return SafeMetaDriveEnv(config)


training_env_3maps = get_training_env({"num_scenarios": 3})
```

You can change the `eval.py` to specify what environment you want to evaluate your agent in:

```python
from env import get_training_env, get_validation_env

# Create the environment
def single_env_factory():
    # validation env:
    return get_validation_env()
    # or training env:
    return get_training_env({"num_scenarios": 3})
```

### **Get Your Agent**

- Train your agent using any RL algorithm in arbitrary environment. You can use `get_training_env()` as a starting
  point.
- Ensure that your agent is properly implemented in the `Policy` class and can run with `eval.py` without bugs (set your
  working directory to `mini_project`).
- Your agent must be self-contained---**do not import files outside your subfolder** or **import any external package
  ** (pytorch and numpy are fine. We will stick to torch=2.6.0). You can refer to `agents/agent_000000000` for an
  example.
- Name your agent subfolder with `agent_YOUR-UID` (e.g., `agents/agent_000000000/`).
- Please make sure your agent can be evaluated by running `python eval.py --agent-name agent_YOUR-UID`, when current working directory is `mini_project`.
- To ensure compatibility, you can create a new conda environment with only torch and some basic packages installed. Then run `eval.py` in this environment to check if your agent can be evaluated.

## **What You Need to Submit**

Your submission must contain:

1. **Your trained RL agent** (The agent zip file)
    - Place your agent in a subfolder under `agents/`.
    - Zip this subfolder, so that after unzip we get a FOLDER instead of files.
    - DO NOT SUBMIT THE AGENTS IN GENERALIZATION EXPERIMENTS.
    - upload it to bruinlearn assignment submission box with name "agent zip file".

2. **A PDF Report**
    - Include your **name and UID**.
    - Describe your **efforts and innovations** in training your agent.
    - Present results from your **generalization experiment** with evaluation plots and analysis.
    - **Submit this PDF inside your agent folder AND upload it to Gradescope.**

3. **Your training code and logs (if any)** (The code zip file)
    - Zip the whole mini project folder, include all your code and logs and upload it to bruinlearn assignment submission box with name "code zip file".

## **Steps to Complete the Assignment**

1. **Select an RL algorithm** to train your agent.
2. **Train your agent** in training environments (with 1, 3, and 10 maps).
3. **Evaluate your agents' generalization** on the corresponding training environments as well as the validation
   environment and analyze its performance.
4. **Refine your agent** to achieve the best performance possible.
5. **Make your submission**.

## **Environment Overview**

The agent needs to steer the target vehicle with low-level acceleration, braking, and steering,
to reach its destination.
Specifically, in MetaDrive safety environments, the agent needs to avoid any crash in the heavy-traffic
scene with normal vehicles, obstacles, and parked vehicles.

In `env.py`, there exists a split of training and validation environments, and we will evaluate your agent in the
validation
environment as well as a hidden test environment. The training/validation/test environments share the same
configuration and the only difference is the random seed that is used to generate the environment (
map, object layout, and traffic flow).

Remember that we only require you to submit one agent and there is no requirement on which environment and which
algorithm
you use to acquire the agent.

You don't have to use the same training environment to get your final agent.

You don't have to avoid using the validation environment for training (as TA can't really know whether you train on
validation env right? so to ensure fairness, you're OK to use the validation environment for training and that's why we
also have a hidden test env).

## Evaluation Protocol & Grading

### Baseline

We will evaluate your agent on the validation environment. Concretely, we will run your agent via this command:

```bash
python eval.py --agent-name agent_YOUR-UID

# You can also play with the provided SB3 PPO agent:
python eval.py --agent-name agent_sb3
```

The above command will run your agent in the validation environment for 100 episodes and report the average episode
reward,
success rate, crash rate, route completion, and other metrics.

**Passing score: 50% route completion rate.**

### Bonus

We will evaluate your agent in the hidden test environment. The hidden test environment is different from the validation
environment in terms of the random seed used to generate the environment and the test environment will be slightly
harder.

### Grading scheme

1. Baseline Report (30%)
    * Generalization Curves (15%): Plot the generalization curves. Analysis of the generalization performance.
    * Method Introduction (15%): Clear and structured explanation of the RL algorithm used, including the reasoning
      behind the choice of architecture, hyperparameters, and training strategies.

2. Performance of your agent in validation Environment (50%)
    * You get 30% if your agent's performance exceeds the passing score threshold.
    * You get 20% for any extra route completion (RC) > 50%. That is you will have `20% * (RC - 0.5)`.

3. Performance in Hidden Test Environment (70%)
    * We will evaluate all submissions and collect the `route_completion` as well as `episode_cost` in test env.
    * We will normalize each `route_completion` and `episode_cost` to range [0, 1] across all students.
        * That is, the normalized RC will be `(RC - RC.min()) / (RC.max() - RC.min())`.
        * The normalized cost will be `(Cost - Cost.min()) / (Cost.max() - Cost.min())`.
    * We will weighted sum the RC and cost to get the final metric: `composite = normRC - alpha * normCost`, `alpha`
      in [0, 1] and will be hidden.
    * We will normalize the composite score and use that to grade `70% * normcomposite` each student's agent.

4. Bonus: Ablation Study (50%)
    * A well-documented analysis of different training strategies, hyperparameters, or architectural choices and their
      impact on performance.
    * You don't need to get the best agent in order to earn this. We want to see your insights and lessons learned from
      experimenting different setup.

5. The maximum scores you can get is 150%.


## Local Verification

You can run the following command to verify your agent locally:

```bash
# Install the required packages
conda env create -f environment.yml

conda activate cs260r_miniproject

python eval.py --agent-name agent_YOUR-UID
```
