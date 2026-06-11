"""
Training harness (model-agnostic): REINFORCE with per-character credit
assignment over a curriculum (train), and supervised teacher-forcing on known
reference programs (sft). Both take the model instance to train.
"""

import os
import random
from collections import deque

import numpy as np
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
          solve_threshold=0.8, window=50, max_places=48, verify_extra=0,
          print_every=10, ckpt_every=500, ckpt_dir="scratch/rl"):
    """
    REINFORCE with per-character credit assignment and a curriculum. Each step:
    sample a target from the unlocked stages, roll out k attempts, score every
    partial program, and give each character its reward-to-go (gain in correct
    leading terms at it and after, minus a gated size penalty) against a
    per-step leave-one-out baseline across attempts. Unlock the next stage
    once a window of recent frontier targets is reliably solved; earlier stages
    stay in rotation as replay.

    With verify_extra > 0, the target is rolled that many terms past what the
    model is shown and scoring uses the full sequence, so printing the shown
    terms literally cannot count as a solve (the continuation exposes it).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    frontier = 0
    recent = deque(maxlen=window)
    for step in range(steps):
        stage = rng.choice(STAGES[:frontier + 1])
        target = sample_target(rng, stage, extra=verify_extra)
        shown = target[:-verify_extra] if verify_extra else target

        # k attempts; for each, score every partial program and build the
        # per-step reward-to-go
        returns, logps_all, entropies, leading = [], [], [], []
        for _s, _status, logps, programs, placed, entropy in (
                train_rollout(model, shown, max_places=max_places)
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


def _sft_update(model, opt, minibatch, chunk, max_places):
    """
    One optimizer step of lockstep teacher-forcing over minibatch, a list of
    (target_sequence, reference_grid) pairs (see sft). Returns the summed
    cross-entropy for logging.
    """
    cross_entropy_sum = nn.CrossEntropyLoss(reduction="sum")
    opt.zero_grad()
    running = 0.0
    for c0 in range(0, len(minibatch), chunk):
        part = minibatch[c0:c0 + chunk]
        tokens = torch.tensor(obs_to_tokens([t for t, _ in part]))
        pad_mask = tokens != PAD
        mem = model.encode_observations(tokens)
        steppers = [Stepper((H, W)) for _ in part]
        active = list(range(len(part)))
        round_losses = []
        for _ in range(max_places):
            active = [i for i in active
                      if steppers[i].run(2000) == "newcell"]
            if not active:
                break
            states = [steppers[i].worldstate() for i in active]
            grids = torch.from_numpy(np.stack([s[0] for s in states]))
            fills = torch.from_numpy(np.stack([s[1] for s in states]))
            ips = torch.tensor([s[2] for s in states])
            heads = torch.tensor([s[3] for s in states])
            toks, pos = model.encode_worldstate_batch(grids, fills, ips, heads)
            idx = torch.tensor(active)
            logits = model.forward_batch(toks, pos, mem[idx], pad_mask[idx])
            ref_ops = [int(part[i][1][s[2][1], s[2][0]])
                       for i, s in zip(active, states)]
            round_losses.append(
                cross_entropy_sum(logits, torch.tensor(ref_ops)))
            for i, op in zip(active, ref_ops):
                steppers[i].place(op)
        chunk_loss = torch.stack(round_losses).sum() / len(minibatch)
        chunk_loss.backward()           # one graph per chunk, then freed
        running += chunk_loss.item()
    opt.step()
    return running


def sft(model, pairs, steps=400, lr=1e-3, seed=0, batch=32, chunk=32,
        max_places=64, print_every=50, ckpt_every=200,
        ckpt_dir="scratch/sft"):
    """
    Supervised fit on (target_sequence, program_source) pairs. Each step samples
    a minibatch of `batch` pairs and teacher-forces the reference programs in
    lockstep: every round, each still-active program's stepper advances to its
    next blank cell, all their worldstates run through one batched forward, the
    op choice is trained toward the reference character (cross-entropy), and
    the reference op is placed. A program leaves the round-loop when its IP
    loops in filled cells. Lockstep keeps token counts identical across active
    programs (round t = t placements + the IP token), so no padding is needed.

    The minibatch is processed in chunks of `chunk` programs, with one backward
    per chunk, bounding peak autograd memory. Requires a model with the batched
    interface (encode_worldstate_batch / forward_batch), i.e. the Transformer.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    refs = [(target, to_grid(src)) for target, src in pairs]

    for step in range(steps):
        minibatch = rng.sample(refs, min(batch, len(refs)))
        running = _sft_update(model, opt, minibatch, chunk, max_places)
        if step % print_every == 0:
            print(f"step {step:4d} CE_loss {running:.4f}")
        if step > 0 and step % ckpt_every == 0:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"sft_step{step}.pt"))
            print(f"saved checkpoint at step {step}")

    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "cnn_sft.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")

    for target, _ in refs[:20]:
        s, status, _trace = rollout(model, target, seed=0)
        n = num_leading(from_grid(s.worldstate()[0]), "befunge", target,
                        max_steps=VERIFY_MAX_STEPS)
        print(f"  target {target}: out={s.output[:24]!r} "
              f"N={n}/{len(target)} status={status}")
    return model


def _world_producer(out_queue, seed):
    """
    Worker loop: generate small batches of fresh random worlds forever and
    push them onto the queue (see sft_stream).
    """
    from models.common.random_worlds import random_worlds
    n = 0
    while True:
        out_queue.put(random_worlds(16, seed=seed + n))
        n += 1


def sft_stream(model, steps=3000, lr=1e-3, seed=0, batch=32, chunk=32,
               max_places=64, producers=4, buffer_cap=2000, eval_pairs=None,
               eval_every=100, eval_k=8, print_every=10, ckpt_every=500,
               ckpt_dir="scratch/stream"):
    """
    Streaming sft: producer processes generate fresh random worlds continuously
    and the trainer samples each minibatch from a rolling buffer of the newest
    ones, so there is no fixed dataset to memorize. Training itself is the same
    lockstep update as sft. Every eval_every steps, runs a quick best-of-eval_k
    reconstruction check on eval_pairs (only their outputs are used).
    """
    import multiprocessing as mp
    import queue as queue_mod

    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    work_queue = mp.Queue(maxsize=64)
    workers = [
        mp.Process(target=_world_producer,
                   args=(work_queue, seed + 1_000_003 * (i + 1)), daemon=True)
        for i in range(producers)
    ]
    for w in workers:
        w.start()

    buffer = deque(maxlen=buffer_cap)
    fresh = 0
    try:
        while len(buffer) < batch:
            buffer.extend(work_queue.get())
        for step in range(steps):
            # drain whatever the producers have ready, without blocking
            while True:
                try:
                    buffer.extend(work_queue.get_nowait())
                    fresh += 16
                except queue_mod.Empty:
                    break
            minibatch = [(t, to_grid(src))
                         for t, src in rng.sample(list(buffer), batch)]
            running = _sft_update(model, opt, minibatch, chunk, max_places)
            if step % print_every == 0:
                print(f"step {step:5d} CE_loss {running:8.4f} "
                      f"buffer {len(buffer)} fresh {fresh}")
            if eval_pairs is not None and step % eval_every == 0:
                eval_reconstruction(model, eval_pairs, k=eval_k,
                                    max_places=max_places)
            if step > 0 and step % ckpt_every == 0:
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"stream_step{step}.pt"))
    finally:
        for w in workers:
            w.terminate()

    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "stream_final.pt")
    torch.save(model.state_dict(), path)
    print(f"saved {path}")
    return model


@torch.no_grad()
def eval_reconstruction(model, pairs, k=32, max_places=64):
    """
    Best-of-k reproduction: for each (output, _) pair, sample k rollouts and
    keep the best leading-term count. Best-of-1 punishes a single bad sample
    in a long program; best-of-k measures whether the policy puts mass near a
    valid program, which is what matters for verifier reranking and as an RL
    start. Prints and returns (full, ge1, avg_leading) for the best-of-k.
    """
    model.eval()
    full = ge1 = total = 0
    for target, _src in pairs:
        n_best = 0
        for sample in range(k):
            s, _status, _trace = rollout(model, target, seed=sample,
                                         max_places=max_places)
            n = num_leading(from_grid(s.worldstate()[0]), "befunge", target,
                            max_steps=VERIFY_MAX_STEPS)
            n_best = max(n_best, n)
            if n_best == len(target):
                break
        full += n_best == len(target)
        ge1 += n_best >= 1
        total += n_best
    m = len(pairs)
    print(f"best-of-{k}: full {full}/{m} | >=1 term {ge1}/{m} | "
          f"avg leading {total / m:.2f}")
    return full, ge1, total / m
