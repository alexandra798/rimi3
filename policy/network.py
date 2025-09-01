"""Alpha挖掘策略网络"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from core import TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, RPNValidator


class PolicyNetwork(nn.Module):
    """策略网络：学习选择下一个Token（包括END）"""

    def __init__(self, state_dim=TOTAL_TOKENS + 3, action_dim=TOTAL_TOKENS, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')

        # GRU编码器（处理Token序列）
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=64,
            num_layers=4,  # 论文指定4层
            batch_first=True,
            dropout=0.1
        )

        # 策略头（输出每个Token的选择概率）
        self.policy_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, action_dim)  # 输出所有Token的logits
        )

    def forward(self, state_encoding, valid_actions_mask=None, lengths=None, return_log_probs=False, return_hidden=False):
        """
        Args:
            state_encoding: [batch_size, seq_len, state_dim] 状态编码
            valid_actions_mask: [batch_size, action_dim] 合法动作掩码
             return_log_probs: 是否返回log概率（供优化器使用）
        Returns:
            action_probs: [batch_size, action_dim] 动作概率分布
            state_value: [batch_size, 1] 状态价值估计
            log_probs: [batch_size, action_dim] 动作对数概率（可选）
        """

        # GRU编码
        gru_out, _ = self.gru(state_encoding)

        if lengths is not None:
            L = gru_out.size(1)
            idx = (lengths - 1).clamp(min=0, max=L - 1)
            last_hidden = gru_out[torch.arange(gru_out.size(0), device=gru_out.device), idx, :]
        else:
            last_hidden = gru_out[:, -1, :]




        action_logits = self.policy_head(last_hidden)

        action_logits = torch.clamp(action_logits, min=-20.0, max=20.0)

        if valid_actions_mask is not None:
            # 非法动作置为 -inf，确保 softmax 后概率严格为 0
            # 注意：不能在这之后再 clamp，否则会把 -inf 复活
            action_logits = action_logits.masked_fill(~valid_actions_mask, float('-inf'))

        # 计算概率（log-softmax 更稳健）
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = torch.exp(log_probs)

        # “全非法行”兜底（避免整行都是 -inf 导致 NaN）
        if valid_actions_mask is not None:
            all_invalid = (~valid_actions_mask).all(dim=-1, keepdim=True)
            if all_invalid.any():
                # 退路：固定把 'END' 的概率设为 1（或您项目里 END 的索引）
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
        根据当前状态选择动作

        Args:
            state: MDPState对象
            temperature: 温度参数，控制探索程度

        Returns:
            action: 选择的Token名称
            action_prob: 该动作的概率
        """
        # 编码状态
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # 获取合法动作
        valid_tokens = RPNValidator.get_valid_next_tokens(state.token_sequence)
        valid_actions_mask = torch.zeros(TOTAL_TOKENS, dtype=torch.bool)
        for token_name in valid_tokens:
            valid_actions_mask[TOKEN_TO_INDEX[token_name]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 直接拿 logits，再统一做 mask/温度/softmax，避免对概率再取 log
            # 复用 forward 的逻辑即可：传入 valid_actions_mask，内部已做掩码与兜底
            action_probs = self.forward(state_encoding, valid_actions_mask)

            if temperature != 1.0:
                # 取对数等效于作用在 logits 上的温度缩放（forward 里已做 log_softmax）
                log_probs = torch.log(action_probs + 1e-12) / temperature
                action_probs = torch.softmax(log_probs, dim=-1)

            # 行内兜底（极罕见场景：数值波动导致全 0）
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