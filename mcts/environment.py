"""MDP环境 - 移除重复代码"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import logging

from core import TOKEN_DEFINITIONS, TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, TokenType, RPNValidator
from alpha import FormulaEvaluator

logger = logging.getLogger(__name__)


class MDPState:
    """MDP环境的状态"""

    def __init__(self):
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.step_count = 0
        self.stack_size = 0

    def add_token(self, token_name):
        if token_name == 'BEG' and self.step_count > 0:
            raise ValueError("BEG can only appear at position 0")

        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.step_count += 1

        # 更新栈大小 —— 以解析器为唯一真源，防止 arity 不一致导致栈错位
        from core import RPNValidator  # 顶部已导入则可去掉此行
        self.stack_size = RPNValidator.calculate_stack_size(self.token_sequence)


    def encode_for_network(self):
        """编码状态用于神经网络输入"""
        max_length = 30
        encoding = np.zeros((max_length, TOTAL_TOKENS + 3))

        for i, token in enumerate(self.token_sequence[:max_length]):
            if i >= max_length:
                break

            token_idx = TOKEN_TO_INDEX[token.name]
            encoding[i, token_idx] = 1
            encoding[i, TOTAL_TOKENS] = i / max_length
            encoding[i, TOTAL_TOKENS + 1] = self.stack_size / 10.0
            encoding[i, TOTAL_TOKENS + 2] = self.step_count / max_length

        return encoding

    def copy(self):
        """深拷贝状态"""
        new_state = MDPState()
        new_state.token_sequence = self.token_sequence.copy()  # Create a shallow copy of the list
        new_state.step_count = self.step_count
        new_state.stack_size = self.stack_size
        return new_state


class AlphaMiningMDP:
    """完整的马尔可夫决策过程环境"""

    def __init__(self):
        self.max_episode_length = 30
        self.current_state = None
        self.formula_evaluator = FormulaEvaluator()  # 使用统一的评估器

    def reset(self):
        """开始新的episode"""
        self.current_state = MDPState()
        return self.current_state

    def step(self, action_token):
        """执行一个动作"""
        if not self.is_valid_action(action_token):
            return self.current_state, -1.0, True

        self.current_state.add_token(action_token)

        if action_token == 'END':
            done = True
        else:
            done = False

        if self.current_state.step_count >= self.max_episode_length:
            done = True

        return self.current_state, 0.0, done

    def is_valid_action(self, action_token):
        """检查动作是否合法"""
        valid_actions = RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)
        return action_token in valid_actions

    def get_valid_actions(self, state):
        """获取当前状态的合法动作 过滤会产生常数的组合"""
        base_actions = RPNValidator.get_valid_next_tokens(state.token_sequence)

        # 过滤掉会产生常数的动作组合
        filtered_actions = []

        for action in base_actions:
            # 检查是否为时序操作符后跟过小的窗口
            if len(state.token_sequence) > 0:
                last_token = state.token_sequence[-1]

                # 如果上一个是时序操作符，检查窗口大小

                if last_token.name in ['ts_skew'] and action == 'delta_3':
                    continue  # 跳过

                if last_token.name in ['ts_kurt'] and action == 'delta_3':
                    continue  # 跳过

                # 检查常数运算
                # 如果栈顶是常数，避免某些操作
                if self.is_stack_top_constant(state):
                    if action in ['ts_std', 'ts_var', 'ts_skew', 'ts_kurt']:
                        continue  # 常数的统计量还是常数

            filtered_actions.append(action)

        # 多样性与确定性约束：限制连续操作数、限制重复操作数
        actions = filtered_actions

        # 限制连续操作数（最近连续≥2个操作数时，只允许操作符或 END）
        consecutive_operands = 0
        for tok in reversed(state.token_sequence):
            if tok.type == TokenType.OPERAND and tok.name != 'BEG':
                consecutive_operands += 1
            else:
                break
        if consecutive_operands >= 2:
            actions = [a for a in actions
                       if (a == 'END') or (TOKEN_DEFINITIONS[a].type == TokenType.OPERATOR)]

        # 限制同一操作数在最近窗口的高频重复（近5步≥3次则临时禁用）
        recent_names = [t.name for t in state.token_sequence[-5:]]
        counts = {}
        for name in recent_names:
            counts[name] = counts.get(name, 0) + 1
        actions = [a for a in actions
                   if not (TOKEN_DEFINITIONS[a].type == TokenType.OPERAND and counts.get(a, 0) >= 3)]

        if not actions:
            # 回退到语法层的原始动作集合（永不发明 END）
            actions = base_actions

        if not actions:
            return []

        return actions

    def is_stack_top_constant(self, state):
        """检查栈顶是否为常数"""
        # 简化实现：检查最近的操作数是否为常数
        for token in reversed(state.token_sequence):
            if token.type == TokenType.OPERAND:
                return token.name.startswith('const_')
            if token.type == TokenType.OPERATOR:
                break
        return False

