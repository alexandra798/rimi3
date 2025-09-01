"""风险寻求策略优化器"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, RPNValidator


class RiskSeekingOptimizer:
    """风险寻求策略优化（优化最好情况而非平均情况）"""

    def __init__(self, policy_network, quantile_alpha=0.85, device=None):
        self.policy_network = policy_network
        self.quantile_alpha = quantile_alpha  # 目标分位数
        self.quantile_estimate = -1.0  # 当前分位数估计
        self.beta = 0.01  # 分位数更新学习率
        self.device = device if device else torch.device('cpu')
        self.gamma = 1.0

        self.optimizer = torch.optim.Adam(
            policy_network.parameters(),
            lr=0.001  # 网络参数学习率
        )

    def update_quantile(self, episode_reward):
        """
        更新分位数估计
        公式：q_{i+1} = q_i + β(1 - α - 1{R(τ_i) ≤ q_i})
        """
        indicator = 1.0 if episode_reward <= self.quantile_estimate else 0.0
        self.quantile_estimate += self.beta * (1 - self.quantile_alpha - indicator)
        return self.quantile_estimate

    def train_on_episode(self, episode_trajectory, gamma=None):
        """
        Args: episode_trajectory: [(state, action, reward), ...]
        Returns: (updated: bool, loss_value: float)
        """
        if gamma is None:
            gamma = self.gamma

        total_reward = sum([r for _, _, r in episode_trajectory])
        self.update_quantile(total_reward)

        # 只有当奖励超过分位数时才更新（风险寻优）
        if total_reward > self.quantile_estimate:
            states_enc, actions_idx, masks, lengths = [], [], [], []

            for state, action, _ in episode_trajectory:
                pre_state = state
                if len(pre_state.token_sequence) >= 2 and pre_state.token_sequence[-1].name == action:
                    pre_state = pre_state.copy()
                    pre_state.token_sequence.pop()
                    pre_state.step_count = max(0, pre_state.step_count - 1)
                    from core import RPNValidator
                    pre_state.stack_size = RPNValidator.calculate_stack_size(pre_state.token_sequence)

                from core import RPNValidator, TOKEN_TO_INDEX
                valid_tokens = RPNValidator.get_valid_next_tokens(pre_state.token_sequence)
                if action not in valid_tokens:
                    import logging
                    logging.debug(f"Skip illegal pair in training: action={action}, valid={valid_tokens}")
                    continue

                states_enc.append(pre_state.encode_for_network())
                actions_idx.append(TOKEN_TO_INDEX[action])

                row_mask = [False] * len(TOKEN_TO_INDEX)
                for name in valid_tokens:
                    row_mask[TOKEN_TO_INDEX[name]] = True
                masks.append(row_mask)
                lengths.append(min(len(pre_state.token_sequence), 30))

            if not states_enc:
                return False, 0.0

            import numpy as np, torch
            states_tensor = torch.as_tensor(np.array(states_enc), dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(actions_idx, dtype=torch.long, device=self.device)
            masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)

            action_probs, log_probs = self.policy_network(
                states_tensor,
                valid_actions_mask=masks_tensor,
                lengths=lengths_tensor,
                return_log_probs=True
            )

            chosen_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
            if not torch.isfinite(chosen_log_probs).all():
                bad = (~torch.isfinite(chosen_log_probs)).nonzero(as_tuple=False).squeeze(-1).tolist()
                raise RuntimeError(f"Chosen log-prob is not finite at indices {bad}")

            # 最大似然：最小化 -sum(log p(a|s))
            loss = -chosen_log_probs.sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            self.optimizer.step()

            return True, float(loss.item())

        return False, 0.0


    def supervised_update(self, batch):
        """
        batch: List of (state_enc, actions_names, pi_vec) for root visit distribution
        目标：最小化 KL(π || pθ)
        """
        import torch, numpy as np
        from core import TOKEN_TO_INDEX
        states, target_probs = [], []

        for state_enc, actions, pi in batch:
            states.append(state_enc)
            # 映射到完整 action 维度
            vec = np.zeros((len(TOKEN_TO_INDEX),), dtype=np.float32)
            for a, p in zip(actions, pi):
                idx = TOKEN_TO_INDEX.get(a, None)
                if idx is not None:
                    vec[idx] = p
            # 避免全零
            s = vec.sum()
            if s <= 0:
                vec[:] = 1.0 / len(vec)
            target_probs.append(vec)

        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(np.array(target_probs), dtype=torch.float32, device=self.device)

        pred_probs, log_probs = self.policy_network(states_tensor, valid_actions_mask=None, return_log_probs=True)

        # KL(π || pθ) = sum π * (log π - log pθ)
        eps = 1e-12
        log_target = torch.log(torch.clamp(target_tensor, min=eps))
        loss = torch.nn.functional.kl_div(log_probs, target_tensor, log_target=False, reduction='batchmean')

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()
        return float(loss.item())
