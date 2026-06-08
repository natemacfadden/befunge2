"""
REINFORCE training for the CNN: sample rollouts, reward by how many leading
terms the generated program reproduces (minus a small size penalty), and push
the policy toward higher-reward programs.
"""

import os
import random
from collections import deque

import torch

from bench.verify import num_leading
from models.cnn.curriculum import STAGES, sample_target
from models.cnn.model import CNN
from models.cnn.rollout import train_rollout
from models.cnn.tokenization import H, W, from_grid

SIZE_PENALTY = 0.5 / (H * W)   # lambda; max total penalty < 1 correct term
VERIFY_MAX_STEPS = 5000        # tight interpreter budget for reward verify


def program_reward(stepper, target):
    """
    Reward for a generated program: N correct leading terms, minus a small
    per-cell size penalty, but only once it gets at least one term right.

    Returns (reward, N, S), with N = correct leading terms and S = placed
    non-blank cells.
    """
    program = from_grid(stepper.worldstate()[0])
    n = num_leading(program, "befunge", target, max_steps=VERIFY_MAX_STEPS)
    s = int((stepper.filled & (stepper.grid != ord(" "))).sum())
    reward = n - (SIZE_PENALTY * s if n >= 1 else 0.0)
    return reward, n, s


def train(steps=5000, k=8, lr=1e-3, seed=0, entropy_coef=0.01,
          solve_threshold=0.8, window=50, max_places=H * W,
          print_every=1, ckpt_every=200, ckpt_dir="checkpoints"):
    """
    REINFORCE with a mean baseline and a curriculum. Each step: sample one
    target from the unlocked stages, roll out k programs, and push the policy
    toward the above-average ones (plus a small entropy bonus). Unlock the next
    stage once a window of recent frontier targets is reliably solved. Earlier
    stages stay in rotation as replay so the policy does not regress on them.

    max_places caps op placements per rollout; the curriculum programs are tiny,
    so a small cap is far faster than filling the whole grid.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    frontier = 0                    # highest unlocked stage index
    recent = deque(maxlen=window)   # solved flags, frontier-stage steps only
    for step in range(steps):
        # sample any unlocked stage, so old families keep getting practiced
        stage = rng.choice(STAGES[:frontier + 1])
        target = sample_target(rng, stage)

        # k rollouts on the same target; reward each
        runs = [train_rollout(model, target, max_places=max_places)
                for _ in range(k)]
        rewards, leading, logprobs, entropies = [], [], [], []
        for stepper, _status, logprob, entropy in runs:
            reward, n, _size = program_reward(stepper, target)
            rewards.append(reward)
            leading.append(n)
            logprobs.append(logprob)
            entropies.append(entropy)

        # advantage = reward - mean baseline; score-function update
        rewards_t = torch.tensor(rewards)
        advantages = rewards_t - rewards_t.mean()
        logprobs_t = torch.stack(logprobs)
        entropy = torch.stack(entropies).mean()
        loss = -(advantages * logprobs_t).mean() - entropy_coef * entropy
        opt.zero_grad()
        loss.backward()
        opt.step()

        # advancement tracks the frontier stage only
        solved = max(leading) == len(target)
        if stage == STAGES[frontier]:
            recent.append(solved)
        solve_rate = sum(recent) / len(recent) if recent else 0.0
        if step % print_every == 0:
            print(f"step {step:5d} stage={stage:10s} "
                  f"meanR={rewards_t.mean():+.3f} bestN={max(leading)} "
                  f"frontier_solve_rate={solve_rate:.2f}")

        # unlock the next stage once the frontier window is reliably solved
        if (len(recent) == window and solve_rate >= solve_threshold
                and frontier < len(STAGES) - 1):
            frontier += 1
            recent.clear()
            print(f"--- unlocking stage '{STAGES[frontier]}' ---")

        # checkpoint semi-often
        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"cnn_step{step}.pt")
            torch.save(model.state_dict(), path)
            print(f"saved {path}")

    return model
