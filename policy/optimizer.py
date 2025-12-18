"""Risk-seeking policy optimizer.

Implements a risk-seeking optimization strategy that focuses on improving
best-case outcomes rather than average performance.
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, RPNValidator


class RiskSeekingOptimizer:
    """Risk-seeking policy optimization.

    Optimizes the policy by suppressing low-performing trajectories,
    instead of maximizing expected (average) return.
    """

    def __init__(self, policy_network, quantile_alpha=0.85, device=None):
        self.policy_network = policy_network
        self.quantile_alpha = quantile_alpha  # Target quantile level (α)
        self.quantile_estimate = 0.0  # Quantile estimate, initialized at 0 instead of -1
        self.beta = 0.05  # Quantile update step size (previously 0.01)
        self.device = device if device else torch.device('cpu')
        self.gamma = 1.0

        self.optimizer = torch.optim.Adam(
            policy_network.parameters(),
            lr=0.001  # Learning rate for network parameters
        )

    def update_quantile(self, episode_reward):
        """
        Update the quantile estimate.

        Formula:
            q_{i+1} = q_i + β (1 - α - 1{R(τ_i) ≤ q_i})

        Args:
            episode_reward: Total reward of the episode trajectory.

        Returns:
            Updated quantile estimate.
        """
        indicator = 1.0 if episode_reward <= self.quantile_estimate else 0.0
        self.quantile_estimate += self.beta * (1 - self.quantile_alpha - indicator)
        return self.quantile_estimate

    def train_on_episode(self, episode_trajectory, gamma=None):
        """
        Train the policy on a single episode trajectory.

        According to Theorem 4.1:
        Only trajectories with total return R ≤ q receive a negative gradient
        (i.e., their probability is pushed down).

        Args:
            episode_trajectory: List of (state, action, reward) tuples.

        Returns:
            (updated: bool, loss_value: float)
        """
        if gamma is None:
            gamma = self.gamma

        # Compute total return of the episode.
        total_reward = sum([r for _, _, r in episode_trajectory])

        # Update the running quantile estimate.
        self.update_quantile(total_reward)

        # Theorem 4.1: apply updates only if R ≤ q; otherwise skip.
        if total_reward > self.quantile_estimate:
            return False, 0.0

        # Prepare training data buffers.
        states_enc, actions_idx, masks, lengths = [], [], [], []

        for state, action, _ in episode_trajectory:
            # Ensure that `state` corresponds to the state *before* taking `action`.
            pre_state = state
            if len(pre_state.token_sequence) >= 2 and pre_state.token_sequence[-1].name == action:
                pre_state = pre_state.copy()
                pre_state.token_sequence.pop()
                pre_state.step_count = max(0, pre_state.step_count - 1)
                from core import RPNValidator
                pre_state.stack_size = RPNValidator.calculate_stack_size(pre_state.token_sequence)

            # Validate action legality under grammar constraints.
            from core import RPNValidator, TOKEN_TO_INDEX
            valid_tokens = RPNValidator.get_valid_next_tokens(pre_state.token_sequence)
            if action not in valid_tokens:
                import logging
                logging.debug(f"Skip illegal pair in training: action={action}, valid={valid_tokens}")
                continue

            # Collect encoded state and chosen action.
            states_enc.append(pre_state.encode_for_network())
            actions_idx.append(TOKEN_TO_INDEX[action])

            # Build a valid-action mask for this state.
            row_mask = [False] * len(TOKEN_TO_INDEX)
            for name in valid_tokens:
                row_mask[TOKEN_TO_INDEX[name]] = True
            masks.append(row_mask)
            lengths.append(min(len(pre_state.token_sequence), 30))

        if not states_enc:
            return False, 0.0

        # Convert collected data to tensors.
        import numpy as np, torch
        states_tensor = torch.as_tensor(np.array(states_enc), dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions_idx, dtype=torch.long, device=self.device)
        masks_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=self.device)

        # Forward pass through the policy network.
        action_probs, log_probs = self.policy_network(
            states_tensor,
            valid_actions_mask=masks_tensor,
            lengths=lengths_tensor,
            return_log_probs=True
        )

        # Extract log-probabilities of the actually chosen actions.
        chosen_log_probs = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Numerical stability check.
        if not torch.isfinite(chosen_log_probs).all():
            bad = (~torch.isfinite(chosen_log_probs)).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise RuntimeError(f"Chosen log-prob is not finite at indices {bad}")

        # Loss: minimize the log-probability of poor trajectories
        # (positive sign because we want to reduce log π, hence suppress probability).
        loss = chosen_log_probs.sum()

        # Backpropagation.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()

        return True, float(loss.item())

    def supervised_update(self, batch):
        """
        Perform a supervised policy update using target action distributions.

        Args:
            batch: List of (state_enc, action_names, pi_vec),
                   where pi_vec is the target distribution from root visit counts.

        Objective:
            Minimize KL(π || p_θ).
        """
        import torch, numpy as np
        from core import TOKEN_TO_INDEX
        states, target_probs = [], []

        for state_enc, actions, pi in batch:
            states.append(state_enc)
            # Map sparse action probabilities into the full action space.
            vec = np.zeros((len(TOKEN_TO_INDEX),), dtype=np.float32)
            for a, p in zip(actions, pi):
                idx = TOKEN_TO_INDEX.get(a, None)
                if idx is not None:
                    vec[idx] = p
            # Avoid an all-zero target distribution.
            s = vec.sum()
            if s <= 0:
                vec[:] = 1.0 / len(vec)
            target_probs.append(vec)

        states_tensor = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(np.array(target_probs), dtype=torch.float32, device=self.device)

        pred_probs, log_probs = self.policy_network(
            states_tensor,
            valid_actions_mask=None,
            return_log_probs=True
        )

        # KL(π || p_θ) = Σ π * (log π - log p_θ)
        eps = 1e-12
        log_target = torch.log(torch.clamp(target_tensor, min=eps))
        loss = torch.nn.functional.kl_div(
            log_probs,
            target_tensor,
            log_target=False,
            reduction='batchmean'
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()
        return float(loss.item())
