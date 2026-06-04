#train

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import DarkForestParallelEnv, A_BROADCAST  


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action="store_true", default=True)
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--tensorboard", action="store_true", default=False)

    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--names", type=str, nargs="+",
                   default=["Santi", "earth", "aliens"],
                   help="civilization names; count = number of agents")
    p.add_argument("--width", type=int, default=10)
    p.add_argument("--height", type=int, default=10)
    p.add_argument("--initial-planets", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--harvest-rate", type=float, default=0.1)
    p.add_argument("--initial-resources", type=float, default=50.0)
    p.add_argument("--initial-population", type=float, default=10.0)
    p.add_argument("--reward", type=str, nargs="*", default=[],
                   help="override env reward weights, e.g. "
                        "--reward broadcast=0 destroyed=50 conquer=3")

    p.add_argument("--total-timesteps", type=int, default=1_000_000,
                   help="env transitions (num_envs*num_steps per iteration)")
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--critic", choices=["independent", "centralized"],
                   default="independent")

    p.add_argument("--stop-mode",
                   choices=["silence", "extermination", "either", "off"],
                   default="silence",
                   help="silence: broadcast rate collapses to ~0 after being "
                        "used; extermination: episodes reliably end with <=1 "
                        "survivor; either: whichever first; off: train to "
                        "--total-timesteps.")
    p.add_argument("--min-iters", type=int, default=30,
                   help="never stop before this many iterations (avoids "
                        "stopping on initial randomness)")
    p.add_argument("--silence-threshold", type=float, default=0.01,
                   help="broadcast-rate EMA below this counts as 'silent'")
    p.add_argument("--silence-patience", type=int, default=10,
                   help="consecutive silent iterations required to stop")
    p.add_argument("--broadcast-peak-threshold", type=float, default=0.05,
                   help="broadcasting must have peaked at least this high, so "
                        "we only stop on *learned* silence, not never-used")
    p.add_argument("--silence-rel-drop", type=float, default=0.25,
                   help="also count as 'silent' if the EMA falls to this "
                        "fraction of its peak (captures collapse without "
                        "needing the rate to literally hit 0)")
    p.add_argument("--annihilation-threshold", type=float, default=0.95,
                   help="fraction of recent episodes ending with <=1 survivor")
    p.add_argument("--ema-beta", type=float, default=0.9)
    return p.parse_args()


class CategoricalMasked(Categorical):
    def __init__(self, logits, masks):
        masks = masks.bool()
        empty = ~masks.any(dim=-1)
        if empty.any():
            masks = masks.clone()
            masks[empty, 0] = True
        self.masks = masks
        neg = torch.finfo(logits.dtype).min
        logits = torch.where(self.masks, logits, torch.full_like(logits, neg))
        super().__init__(logits=logits)

    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.zeros_like(p_log_p))
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, state_dim, n_agents,
                 critic_mode, hidden=256):
        super().__init__()
        self.critic_mode = critic_mode
        self.n_agents = n_agents

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )
        critic_in = (state_dim + n_agents) if critic_mode == "centralized" else obs_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_in, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def get_value(self, obs, state, onehot):
        if self.critic_mode == "centralized":
            return self.critic(torch.cat([state, onehot], dim=-1))
        return self.critic(obs)

    def get_action_and_value(self, obs, mask, state, onehot, action=None):
        logits = self.actor(obs)
        dist = CategoricalMasked(logits, mask)
        if action is None:
            action = dist.sample()
        return (action, dist.log_prob(action), dist.entropy(),
                self.get_value(obs, state, onehot).squeeze(-1))


class MAVecEnv:
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        e0 = self.envs[0]
        self.possible_agents = list(e0.possible_agents)
        self.N = len(self.possible_agents)
        self._slot = {n: i for i, n in enumerate(self.possible_agents)}

        msp = e0.observation_space(self.possible_agents[0])
        c, h, w = msp["map"].shape
        self.map_size = c * h * w
        self.self_size = msp["self"].shape[0]
        self.obs_dim = self.map_size + self.self_size
        self.action_dim = e0.action_dim
        self.state_dim = e0.state_space.shape[0]

        self._placeholder_mask = np.zeros(self.action_dim, dtype=np.float32)
        self._placeholder_mask[0] = 1.0  # explore-only -> always finite
        self._cur = [None] * self.num_envs   # name -> obs dict (alive only)
        self._ep_len = [0] * self.num_envs

    def _flat_obs(self, o):
        return np.concatenate([o["map"].ravel(), o["self"]]).astype(np.float32)

    def _batch(self):
        obs = np.zeros((self.num_envs, self.N, self.obs_dim), dtype=np.float32)
        mask = np.zeros((self.num_envs, self.N, self.action_dim), dtype=np.float32)
        state = np.zeros((self.num_envs, self.state_dim), dtype=np.float32)
        active = np.zeros((self.num_envs, self.N), dtype=np.float32)
        for i, e in enumerate(self.envs):
            state[i] = e.state().astype(np.float32)
            for name, slot in self._slot.items():
                o = self._cur[i].get(name)
                if o is None:
                    mask[i, slot] = self._placeholder_mask
                else:
                    obs[i, slot] = self._flat_obs(o)
                    mask[i, slot] = o["action_mask"].astype(np.float32)
                    active[i, slot] = 1.0
        return obs, mask, state, active

    def reset(self, seed=None):
        for i, e in enumerate(self.envs):
            s = None if seed is None else seed + i
            o, _ = e.reset(seed=s)
            self._cur[i] = dict(o)
            self._ep_len[i] = 0
        obs, mask, state, active = self._batch()
        return obs, mask, state, active

    def step(self, actions):
        rewards = np.zeros((self.num_envs, self.N), dtype=np.float32)
        dones = np.zeros((self.num_envs, self.N), dtype=np.float32)
        infos = [dict() for _ in range(self.num_envs)]
        for i, e in enumerate(self.envs):
            alive = list(e.agents)
            act = {n: int(actions[i, self._slot[n]]) for n in alive}
            obs, rew, term, trunc, _ = e.step(act)
            self._ep_len[i] += 1
            for n in rew:
                slot = self._slot[n]
                rewards[i, slot] = rew[n]
                dones[i, slot] = float(term[n] or trunc[n])
            if len(e.agents) == 0:
                survivors = sum(1 for c in e.civs.values() if c.alive)
                infos[i] = {"episode_end": True,
                            "survivors": survivors,
                            "annihilation": survivors <= 1,
                            "length": self._ep_len[i]}
                o2, _ = e.reset()
                self._cur[i] = dict(o2)
                self._ep_len[i] = 0
            else:
                self._cur[i] = {n: obs[n] for n in e.agents}
        nobs, nmask, nstate, nactive = self._batch()
        return nobs, nmask, nstate, rewards, dones, nactive, infos


def make_env_fn(args, idx):
    def thunk():
        return DarkForestParallelEnv(
            names=args.names, width=args.width, height=args.height,
            initial_planets=args.initial_planets, max_steps=args.max_steps,
            harvest_rate=args.harvest_rate,
            initial_resources=args.initial_resources,
            initial_population=args.initial_population,
            reward_weights=args.reward_weights or None,
        )
    return thunk


class DarkForestStopper:
    def __init__(self, args):
        self.mode = args.stop_mode
        self.min_iters = args.min_iters
        self.sil_thr = args.silence_threshold
        self.sil_patience = args.silence_patience
        self.peak_thr = args.broadcast_peak_threshold
        self.rel_drop = args.silence_rel_drop
        self.ann_thr = args.annihilation_threshold
        self.beta = args.ema_beta
        self.ema = None
        self.peak = 0.0
        self.silent_streak = 0
        self.recent_ann = deque(maxlen=50)

    def update(self, it, broadcast_rate, episode_infos):
        self.ema = (broadcast_rate if self.ema is None
                    else self.beta * self.ema + (1 - self.beta) * broadcast_rate)
        self.peak = max(self.peak, self.ema)
        collapsed = self.peak >= self.peak_thr and self.ema <= self.rel_drop * self.peak
        silent_now = (self.ema < self.sil_thr) or collapsed
        self.silent_streak = self.silent_streak + 1 if silent_now else 0
        for info in episode_infos:
            if info.get("episode_end"):
                self.recent_ann.append(1.0 if info["annihilation"] else 0.0)

        if self.mode == "off" or it < self.min_iters:
            return None

        silence = (self.peak >= self.peak_thr
                   and self.silent_streak >= self.sil_patience)
        ann_rate = np.mean(self.recent_ann) if self.recent_ann else 0.0
        extermination = (len(self.recent_ann) >= self.recent_ann.maxlen
                         and ann_rate >= self.ann_thr)

        if self.mode == "silence" and silence:
            return (f"broadcast silence (EMA={self.ema:.4f} for "
                    f"{self.silent_streak} iters; peaked at {self.peak:.3f})")
        if self.mode == "extermination" and extermination:
            return f"extermination ({ann_rate:.0%} of recent episodes annihilated)"
        if self.mode == "either" and (silence or extermination):
            return ("broadcast silence" if silence
                    else f"extermination ({ann_rate:.0%})")
        return None

def main():
    args = parse_args()
    args.reward_weights = {}
    for kv in args.reward:
        k, v = kv.split("=")
        args.reward_weights[k.strip()] = float(v)
    args.batch_size = args.num_envs * args.num_steps
    run_name = args.run_name or f"darkforest_{args.critic}_{int(time.time())}"

    random_seed(args.seed, args.torch_deterministic)
    print(f"[setup] run={run_name} critic={args.critic} "
          f"stop-mode={args.stop_mode}")

    writer = None
    if args.tensorboard:
        try:
            writer = SummaryWriter(f"runs/{run_name}")
        except Exception as ex:  # noqa: BLE001
            print(f"[warn] tensorboard unavailable ({ex}); continuing without it")

    vec = MAVecEnv([make_env_fn(args, i) for i in range(args.num_envs)])
    N, E = vec.N, vec.num_envs
    batch = E * N                          # rows per timestep (env x agent)
    agent = Agent(vec.obs_dim, vec.action_dim, vec.state_dim, N,
                  args.critic, args.hidden_dim)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    onehot_all = torch.zeros(batch, N)
    for i in range(E):
        for a in range(N):
            onehot_all[i * N + a, a] = 1.0

    T = args.num_steps
    obs_b = torch.zeros((T, batch, vec.obs_dim))
    mask_b = torch.zeros((T, batch, vec.action_dim))
    state_b = torch.zeros((T, batch, vec.state_dim))
    act_b = torch.zeros((T, batch), dtype=torch.long)
    logp_b = torch.zeros((T, batch))
    val_b = torch.zeros((T, batch))
    rew_b = torch.zeros((T, batch))
    done_b = torch.zeros((T, batch))
    active_b = torch.zeros((T, batch))

    def to_rows(arr):  # (E, N, ...) -> (E*N, ...)
        return arr.reshape(batch, *arr.shape[2:])

    obs_np, mask_np, state_np, active_np = vec.reset(seed=args.seed)
    next_obs = torch.tensor(to_rows(obs_np))
    next_mask = torch.tensor(to_rows(mask_np))
    next_state = torch.tensor(np.repeat(state_np, N, axis=0))
    next_done = torch.zeros(batch)
    next_active = torch.tensor(active_np.reshape(batch))

    stopper = DarkForestStopper(args)
    ep_return = np.zeros(batch, dtype=np.float64)
    return_hist = deque(maxlen=100)
    surv_hist = deque(maxlen=100)
    global_step = 0
    num_iters = args.total_timesteps // args.batch_size
    start = time.time()
    stop_reason = None

    for it in range(1, num_iters + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1 - (it - 1) / num_iters) * args.learning_rate

        n_broadcast = 0.0
        n_active = 0.0
        ep_infos = []

        for step in range(T):
            global_step += E
            obs_b[step] = next_obs
            mask_b[step] = next_mask
            state_b[step] = next_state
            done_b[step] = next_done
            active_b[step] = next_active

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(
                    next_obs, next_mask, next_state, onehot_all)
            val_b[step] = value
            act_b[step] = action
            logp_b[step] = logp

            act_np = action.cpu().numpy().reshape(E, N)
            n_broadcast += float(((act_np == A_BROADCAST) & (active_np > 0)).sum())
            n_active += float(active_np.sum())

            obs_np, mask_np, state_np, rew_np, done_np, active_np, infos = vec.step(act_np)
            rew_b[step] = torch.tensor(rew_np.reshape(batch))

            ep_return += rew_np.reshape(batch)
            done_flat = done_np.reshape(batch)
            for r in range(batch):
                if done_flat[r] > 0:
                    return_hist.append(ep_return[r])
                    ep_return[r] = 0.0
            for info in infos:
                if info.get("episode_end"):
                    surv_hist.append(info["survivors"])
                    ep_infos.append(info)

            next_obs = torch.tensor(to_rows(obs_np))
            next_mask = torch.tensor(to_rows(mask_np))
            next_state = torch.tensor(np.repeat(state_np, N, axis=0))
            next_done = torch.tensor(done_flat, dtype=torch.float32)
            next_active = torch.tensor(active_np.reshape(batch))
        # ---- GAE -------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_state, onehot_all).squeeze(-1)
            adv = torch.zeros_like(rew_b)
            lastgae = torch.zeros(batch)
            for t in reversed(range(T)):
                if t == T - 1:
                    nonterminal = 1.0 - next_done
                    nextval = next_value
                else:
                    nonterminal = 1.0 - done_b[t + 1]
                    nextval = val_b[t + 1]
                delta = rew_b[t] + args.gamma * nextval * nonterminal - val_b[t]
                adv[t] = lastgae = delta + args.gamma * args.gae_lambda * nonterminal * lastgae
            returns = adv + val_b

        # ---- flatten & mask inactive rows ------------------------------
        b_obs = obs_b.reshape(-1, vec.obs_dim)
        b_mask = mask_b.reshape(-1, vec.action_dim)
        b_state = state_b.reshape(-1, vec.state_dim)
        b_onehot = onehot_all.repeat(T, 1)
        b_act = act_b.reshape(-1)
        b_logp = logp_b.reshape(-1)
        b_val = val_b.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_active = active_b.reshape(-1)

        live_idx = torch.nonzero(b_active > 0, as_tuple=False).squeeze(-1)
        if live_idx.numel() == 0:
            continue
        mb_size = max(1, live_idx.numel() // args.num_minibatches)

        approx_kl = torch.tensor(0.0)
        for _ in range(args.update_epochs):
            perm = live_idx[torch.randperm(live_idx.numel())]
            for s in range(0, perm.numel(), mb_size):
                mb = perm[s:s + mb_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_state[mb], b_onehot[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_adv = b_adv[mb]
                if args.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                if args.clip_vloss:
                    v_unclipped = (newval - b_ret[mb]) ** 2
                    v_clipped = b_val[mb] + torch.clamp(
                        newval - b_val[mb], -args.clip_coef, args.clip_coef)
                    v_clipped = (v_clipped - b_ret[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((newval - b_ret[mb]) ** 2).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- metrics + dark-forest stop check --------------------------
        broadcast_rate = (n_broadcast / n_active) if n_active > 0 else 0.0
        mean_ret = float(np.mean(return_hist)) if return_hist else float("nan")
        mean_surv = float(np.mean(surv_hist)) if surv_hist else float("nan")
        sps = int(global_step / (time.time() - start))

        if it % 5 == 0 or it == 1:
            print(f"[iter {it:4d}/{num_iters}] step={global_step} "
                  f"broadcast_rate={broadcast_rate:.4f} ema={stopper.ema or 0:.4f} "
                  f"ep_ret={mean_ret:.2f} survivors={mean_surv:.2f} "
                  f"value_loss={float(v_loss):.3f} SPS={sps}")
        if writer:
            writer.add_scalar("charts/broadcast_rate", broadcast_rate, global_step)
            writer.add_scalar("charts/broadcast_ema", stopper.ema or 0, global_step)
            writer.add_scalar("charts/episodic_return", mean_ret, global_step)
            writer.add_scalar("charts/mean_survivors", mean_surv, global_step)
            writer.add_scalar("losses/value_loss", float(v_loss), global_step)
            writer.add_scalar("losses/policy_loss", float(pg_loss), global_step)
            writer.add_scalar("losses/entropy", float(ent_loss), global_step)
            writer.add_scalar("losses/approx_kl", float(approx_kl), global_step)

        stop_reason = stopper.update(it, broadcast_rate, ep_infos)
        if stop_reason:
            print(f"\n[STOP @ iter {it}] dark-forest criterion met: {stop_reason}")
            break

    os.makedirs("runs", exist_ok=True)
    ckpt = f"runs/{run_name}.pt"
    torch.save({"agent": agent.state_dict(), "args": vars(args),
                "stopped": stop_reason, "global_step": global_step}, ckpt)
    print(f"[done] saved {ckpt} after {global_step} steps "
          f"({'stopped: ' + stop_reason if stop_reason else 'reached total-timesteps'})")
    if writer:
        writer.close()
    vec_close(vec)


def vec_close(vec):
    for e in vec.envs:
        e.close()


def random_seed(seed, deterministic):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


if __name__ == "__main__":
    main()