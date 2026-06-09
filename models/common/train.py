"""
Training harness (model-agnostic): REINFORCE with per-character credit
assignment over a curriculum (train), and supervised teacher-forcing on known
reference programs (sft). Both take the model instance to train.
"""

import os
import random
from collections import deque

import torch
import torch.nn as nn

from bench.verify import num_leading
from models.common.curriculum import STAGES, sample_target
from models.common.rollout import rollout, train_rollout
from models.common.stepper import Stepper
from models.common.tokenization import (
    PAD,
    H,
    W,
    from_grid,
    obs_to_tokens,
    to_grid,
)

SIZE_PENALTY = 0.5 / (H * W)   # lambda; max total size penalty < 1 correct term
VERIFY_MAX_STEPS = 5000        # tight interpreter budget for reward verify


def train(model, steps=1000, k=8, lr=1e-3, seed=0, entropy_coef=0.0,
          solve_threshold=0.8, window=50, max_places=48,
          print_every=10, ckpt_every=500, ckpt_dir="checkpoints"):
    """
    REINFORCE with per-character credit assignment and a curriculum. Each step:
    sample a target from the unlocked stages, roll out k attempts, score every
    partial program, and give each character its reward-to-go (gain in correct
    leading terms at it and after, minus a gated size penalty) against a
    per-step leave-one-out baseline across attempts. Unlock the next stage
    once a window of recent frontier targets is reliably solved; earlier stages
    stay in rotation as replay.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    frontier = 0
    recent = deque(maxlen=window)
    for step in range(steps):
        stage = rng.choice(STAGES[:frontier + 1])
        target = sample_target(rng, stage)

        # k attempts; for each, score every partial program and build the
        # per-step reward-to-go
        returns, logps_all, entropies, leading = [], [], [], []
        for _s, _status, logps, programs, placed, entropy in (
                train_rollout(model, target, max_places=max_places)
                for _ in range(k)):
            ns = [num_leading(p, "befunge", target, max_steps=VERIFY_MAX_STEPS)
                  for p in programs]
            n_final = ns[-1] if ns else 0
            gate = 1.0 if n_final >= 1 else 0.0
            rewards, prev = [], 0
            for nt, pl in zip(ns, placed):
                rewards.append((nt - prev) - SIZE_PENALTY * pl * gate)
                prev = nt
            togo, acc = [], 0.0
            for r in reversed(rewards):
                acc += r
                togo.append(acc)
            togo.reverse()
            returns.append(togo)
            logps_all.append(logps)
            entropies.append(entropy)
            leading.append(n_final)

        # per-step leave-one-out baseline across attempts (aligned by placement
        # index), then advantage = reward-to-go - baseline
        width = max((len(g) for g in returns), default=0)
        col_sum, col_cnt = [0.0] * width, [0] * width
        for g in returns:
            for t, v in enumerate(g):
                col_sum[t] += v
                col_cnt[t] += 1
        terms = []
        for g, logps in zip(returns, logps_all):
            for t, (v, lp) in enumerate(zip(g, logps)):
                base = (col_sum[t] - v) / (col_cnt[t] - 1) if col_cnt[t] > 1 \
                    else 0.0
                terms.append((v - base) * lp)

        policy_loss = (-torch.stack(terms).sum() / k if terms
                       else torch.zeros(()))
        ent_mean = torch.stack(entropies).mean() if entropies \
            else torch.zeros(())
        loss = policy_loss - entropy_coef * ent_mean
        opt.zero_grad()
        loss.backward()
        opt.step()

        solved = bool(leading) and max(leading) == len(target)
        if stage == STAGES[frontier]:
            recent.append(solved)
        solve_rate = sum(recent) / len(recent) if recent else 0.0
        if step % print_every == 0:
            mean_r = sum((g[0] if g else 0.0) for g in returns) / max(k, 1)
            best_n = max(leading) if leading else 0
            print(f"step {step:5d} stage={stage:8s} meanR={mean_r:+.3f} "
                  f"bestN={best_n} frontier_solve_rate={solve_rate:.2f}")

        if (len(recent) == window and solve_rate >= solve_threshold
                and frontier < len(STAGES) - 1):
            frontier += 1
            recent.clear()
            print(f"--- unlocking stage '{STAGES[frontier]}' ---")

        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            path = os.path.join(ckpt_dir, f"cnn_step{step}.pt")
            torch.save(model.state_dict(), path)
            print(f"saved {path}")

    return model


def sft(model, pairs, steps=400, lr=1e-3, seed=0, print_every=50,
        ckpt_dir="checkpoints_sft"):
    """
    Supervised fit on (target_sequence, program_source) pairs. Teacher-force
    each reference program through the stepper: at every cell the IP lands on,
    train
    the op choice toward the reference character (cross-entropy) and place it,
    stopping once the IP loops in filled cells. Handles looping programs (e.g.
    the g/p Fibonacci) whose path runs over no-op spaces that must be placed.
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cross_entropy = nn.CrossEntropyLoss()
    refs = [(target, to_grid(src)) for target, src in pairs]

    for step in range(steps):
        example_losses = []
        for target, ref in refs:
            tokens = torch.tensor(obs_to_tokens([target]))
            pad_mask = tokens != PAD
            mem = model.encode_observations(tokens)
            s = Stepper((H, W))
            losses = []
            while len(losses) < H * W:
                if s.run(2000) != "newcell":
                    break
                vocab_grid, filled, (x, y), heading = s.worldstate()
                ws = model.encode_worldstate(
                    torch.tensor(vocab_grid)[None], torch.tensor(filled)[None],
                    torch.tensor([[x, y]]), torch.tensor([heading]))
                logits = model(ws, mem, pad_mask, torch.tensor([[x, y]]))
                ref_op = int(ref[y, x])
                losses.append(cross_entropy(logits, torch.tensor([ref_op])))
                s.place(ref_op)
            example_losses.append(torch.stack(losses).sum())

        loss = torch.stack(example_losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % print_every == 0:
            print(f"step {step:4d} CE_loss {loss.item():.4f}")

    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "cnn_sft.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")

    for target, _ in refs:
        s, status, _trace = rollout(model, target, seed=0)
        n = num_leading(from_grid(s.worldstate()[0]), "befunge", target,
                        max_steps=VERIFY_MAX_STEPS)
        print(f"  target {target}: out={s.output[:24]!r} "
              f"N={n}/{len(target)} status={status}")
    return model
