"""PPO compact (clip surrogate) pour la phase 2 de la spec.

Implementation autonome en PyTorch, sans dependance a stable-baselines3/FinRL.
La politique partage l'architecture deux branches du modele d'imitation et peut
etre initialisee depuis les poids pre-entraines (ScalpingMultimodalNet), avec
une tete valeur ajoutee pour l'actor-critic.
"""

import numpy as np
import torch
import torch.nn as nn

from model_pytorch import ScalpingMultimodalNet


class ActorCritic(nn.Module):
    def __init__(self, order_book_dim, sequence_dim, hidden_dim=64, base_model=None):
        super().__init__()
        if base_model is not None:
            self.backbone = base_model
            self.action_head = nn.Linear(hidden_dim, 3)
            nn.init.zeros_(self.action_head.weight)
            nn.init.zeros_(self.action_head.bias)
        else:
            self.backbone = ScalpingMultimodalNet(
                order_book_dim=order_book_dim, sequence_dim=sequence_dim, hidden_dim=hidden_dim
            )
            self.action_head = nn.Linear(hidden_dim, 3)
        self.value_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, book, seq):
        raise NotImplementedError

    def features(self, book, seq):
        out_book = self.backbone.order_book_branch(book)
        lstm_out, _ = self.backbone.lstm_branch(seq)
        out_seq = self.backbone.lstm_fc(lstm_out[:, -1, :])
        return torch.cat((out_book, out_seq), dim=1)

    def act(self, book, seq):
        feats = self.features(book, seq)
        logits = self.action_head(self.backbone.fusion_layer[0](feats))
        value = self.value_head(feats).squeeze(-1)
        return logits, value

    def policy_logits(self, book, seq):
        feats = self.features(book, seq)
        return self.action_head(self.backbone.fusion_layer[0](feats))


class PPOAgent:
    def __init__(
        self,
        env,
        order_book_dim=None,
        sequence_dim=None,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        epochs_per_update=4,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
        device="cpu",
        seed=42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = env
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        obs = env.observation_space
        self.order_book_dim = order_book_dim or obs["book"][0]
        self.sequence_dim = sequence_dim or obs["seq"][1]
        self.policy = ActorCritic(self.order_book_dim, self.sequence_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _obs_to_tensors(self, obs):
        book = torch.tensor(np.asarray(obs["book"]), dtype=torch.float32).unsqueeze(0).to(self.device)
        seq = torch.tensor(np.asarray(obs["seq"]), dtype=torch.float32).unsqueeze(0).to(self.device)
        return book, seq

    def collect_rollout(self, n_steps):
        """Collecte n_steps transitions avec les probas/log-probas courantes."""
        book_buf, seq_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
        obs = self.current_obs
        for _ in range(n_steps):
            with torch.no_grad():
                book, seq = self._obs_to_tensors(obs)
                logits, value = self.policy.act(book, seq)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            act_buf.append(int(action.item()))
            logp_buf.append(float(logp.item()))
            val_buf.append(float(value.item()))
            book_buf.append(obs["book"])
            seq_buf.append(obs["seq"])
            next_obs, reward, done, _ = self.env.step(int(action.item()))
            rew_buf.append(reward)
            done_buf.append(done)
            obs = self.env.reset() if done else next_obs
        self.current_obs = obs
        return {
            "book": np.array(book_buf),
            "seq": np.array(seq_buf),
            "actions": np.array(act_buf),
            "logp": np.array(logp_buf),
            "values": np.array(val_buf),
            "rewards": np.array(rew_buf, dtype=np.float32),
            "dones": np.array(done_buf),
        }

    def _compute_gae(self, rewards, values, dones):
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(n)):
            if dones[t]:
                next_nonterminal = 0.0
                last_gae = 0.0
            else:
                next_nonterminal = 1.0
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, rollout):
        adv, ret = self._compute_gae(rollout["rewards"], rollout["values"], rollout["dones"])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(adv)
        book = torch.tensor(rollout["book"], dtype=torch.float32).to(self.device)
        seq = torch.tensor(rollout["seq"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.long).to(self.device)
        old_logp = torch.tensor(rollout["logp"], dtype=torch.float32).to(self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32).to(self.device)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_updates = 0
        for _ in range(self.epochs_per_update):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                logits, value = self.policy.act(book[batch], seq[batch])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(actions[batch])
                ratio = torch.exp(logp - old_logp[batch])
                policy_loss = -torch.min(
                    ratio * adv_t[batch],
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_t[batch],
                ).mean()
                value_loss = nn.functional.mse_loss(value, ret_t[batch])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                n_updates += 1
        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats

    def train(self, n_iterations=50, n_steps_per_rollout=512, verbose_every=10):
        self.current_obs = self.env.reset()
        history = []
        for it in range(1, n_iterations + 1):
            rollout = self.collect_rollout(n_steps_per_rollout)
            stats = self.update(rollout)
            stats["mean_reward"] = float(rollout["rewards"].mean())
            stats["capital"] = self.env.capital
            history.append(stats)
            if verbose_every and (it % verbose_every == 0 or it == 1):
                print(
                    f"PPO iter {it:>4d} | reward moyen={stats['mean_reward']:>8.4f} "
                    f"| policy_loss={stats['policy_loss']:.4f} | value_loss={stats['value_loss']:.4f} "
                    f"| entropy={stats['entropy']:.4f} | capital={self.env.capital:.2f}"
                )
        return history

    def save(self, path="ppo_scalping_policy.pth"):
        torch.save({"policy_state_dict": self.policy.state_dict()}, path)
        print(f"Politique PPO sauvegardee sous '{path}'")
