"""Alpha-mining policy network.

Defines a policy network that learns to select the next token (including 'END')
under validity constraints (e.g., RPN grammar constraints).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, RPNValidator


class PolicyNetwork(nn.Module):
    """Policy network: learns to choose the next token (including 'END')."""

    def __init__(self, state_dim=TOTAL_TOKENS + 3, action_dim=TOTAL_TOKENS, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')

        # GRU encoder for token sequences.
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=64,
            num_layers=4,  # Paper specifies 4 layers.
            batch_first=True,
            dropout=0.1
        )

        # Policy head: outputs logits for choosing each token.
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, action_dim)  # Logits over all tokens.
        )

    def forward(self, state_encoding, valid_actions_mask=None, lengths=None, return_log_probs=False, return_hidden=False):
        """
        Args:
            state_encoding: Tensor of shape [batch_size, seq_len, state_dim], encoded state sequence.
            valid_actions_mask: Optional bool mask of shape [batch_size, action_dim], True for valid actions.
            lengths: Optional tensor/list of sequence lengths. If provided, the last valid timestep is used.
            return_log_probs: Whether to additionally return log-probabilities (useful for optimizers).
            return_hidden: Whether to additionally return the last hidden representation used by the policy head.

        Returns:
            If return_log_probs is False:
                action_probs: Tensor [batch_size, action_dim], probability distribution over actions.
                (optionally) last_hidden if return_hidden is True.
            If return_log_probs is True:
                (action_probs, log_probs):
                    action_probs: Tensor [batch_size, action_dim]
                    log_probs: Tensor [batch_size, action_dim]
                (optionally) last_hidden if return_hidden is True.

        Notes:
            - When valid_actions_mask is provided, invalid actions are forced to probability 0.
            - A safety fallback handles the rare case where an entire row is invalid (all masked).
        """

        # GRU encoding.
        gru_out, _ = self.gru(state_encoding)

        # Select the representation at the last valid timestep (if lengths provided),
        # otherwise use the final timestep.
        if lengths is not None:
            L = gru_out.size(1)
            idx = (lengths - 1).clamp(min=0, max=L - 1)
            last_hidden = gru_out[torch.arange(gru_out.size(0), device=gru_out.device), idx, :]
        else:
            last_hidden = gru_out[:, -1, :]




        action_logits = self.policy_head(last_hidden)

        # Keep logits in a reasonable numeric range (except -inf from masking).
        action_logits = torch.clamp(action_logits, min=-20.0, max=20.0)

        if valid_actions_mask is not None:
            # Set invalid actions to -inf so that softmax yields exactly 0 probability.
            # Important: do NOT clamp after this, or -inf would be turned back into a finite value.
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))

        # Compute probabilities (log-softmax is numerically more stable).
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = torch.exp(log_probs)

        # Fallback for "all-invalid rows" (prevents NaNs when a full row becomes -inf).
        if valid_actions_mask is not None:
            all_invalid = (~valid_actions_mask).all(dim=-1, keepdim=True)
            if all_invalid.any():
                # Escape hatch: force 'END' to have probability 1 (or your project's END index).
                end_idx = TOKEN_TO_INDEX.get('END', 0)
                fallback = torch.zeros_like(action_probs)
                fallback[..., end_idx] = 1.0
                action_probs = torch.where(all_invalid, fallback, action_probs)

                fallback_log = torch.full_like(log_probs, float('-inf'))
                fallback_log[..., end_idx] = 0.0
                log_probs = torch.where(all_invalid, fallback_log, log_probs)

        if return_log_probs:
            if return_hidden:
                return (action_probs, log_probs), last_hidden
            return action_probs, log_probs
        else:
            if return_hidden:
                return action_probs, last_hidden
            return action_probs

    def get_action(self, state, temperature=1.0):
        """
        Select an action (token) given the current state.

        Args:
            state: An MDPState-like object with:
                - token_sequence: the current token sequence
                - encode_for_network(): returns a [seq_len, state_dim] encoding
            temperature: Exploration control. 1.0 means no scaling.
                         Higher -> more random; lower -> more greedy.

        Returns:
            action: Selected token name (string).
            action_prob: Probability assigned to that token (float).
        """
        # Encode the current state into a batched tensor.
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # Build a boolean mask over valid actions based on the current token sequence.
        valid_tokens = RPNValidator.get_valid_next_tokens(state.token_sequence)
        valid_actions_mask = torch.zeros(TOTAL_TOKENS, dtype=torch.bool)
        for token_name in valid_tokens:
            valid_actions_mask[TOKEN_TO_INDEX[token_name]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Prefer working from logits with consistent masking/softmax behavior.
            # Reuse forward(): it already applies the mask and includes a safety fallback.
            action_probs = self.forward(state_encoding, valid_actions_mask)

            if temperature != 1.0:
                # Temperature scaling: operate in log space to emulate scaling logits.
                # forward() already produced a normalized distribution, so convert to log safely.
                log_probs = torch.log(action_probs + 1e-12) / temperature
                action_probs = torch.softmax(log_probs, dim=-1)

            # Row-level fallback (extremely rare): numeric issues yield an all-zero row.
            row_sum = action_probs.sum(dim=-1, keepdim=True)
            zero_row = (row_sum <= 0)
            if zero_row.any():
                end_idx = TOKEN_TO_INDEX.get('END', 0)
                fallback = torch.zeros_like(action_probs)
                fallback[..., end_idx] = 1.0
                action_probs = torch.where(zero_row, fallback, action_probs)

            action_idx = torch.multinomial(action_probs[0], 1).item()
            action_name = INDEX_TO_TOKEN[action_idx]
            action_prob = action_probs[0, action_idx].item()

        return action_name, action_prob
