"""
Markov Decision Process (MDP) environment for alpha mining.

This module defines:
- The MDP state representation used by MCTS
- The environment dynamics (state transition, action validation)
- Domain-specific constraints to avoid trivial or degenerate formulas
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import logging

from core import TOKEN_DEFINITIONS, TOKEN_TO_INDEX, INDEX_TO_TOKEN, TOTAL_TOKENS, TokenType, RPNValidator
from alpha import FormulaEvaluator

logger = logging.getLogger(__name__)


class MDPState:
    """
    State representation for the MDP.

    The state tracks:
    - The current token sequence (in Reverse Polish Notation)
    - The number of steps taken so far
    - The current stack size implied by the token sequence
    """

    def __init__(self):
        # Token sequence always starts with a BEG token
        self.token_sequence = [TOKEN_DEFINITIONS['BEG']]
        self.step_count = 0
        self.stack_size = 0

    def add_token(self, token_name):
        """
        Append a token to the current token sequence and update state metadata.

        """
        # BEG is only allowed at the initial position
        if token_name == 'BEG' and self.step_count > 0:
            raise ValueError("BEG can only appear at position 0")

        token = TOKEN_DEFINITIONS[token_name]
        self.token_sequence.append(token)
        self.step_count += 1

        # Update stack size.
        # The RPNValidator is treated as the single source of truth to avoid
        # inconsistencies caused by incorrect operator arity assumptions.
        from core import RPNValidator
        self.stack_size = RPNValidator.calculate_stack_size(self.token_sequence)


    def encode_for_network(self):
        """
        Encode the current state into a fixed-size tensor for neural network input.

        Encoding includes:
        - One-hot token identity
        - Normalized token position
        - Normalized stack size
        - Normalized step count
        """
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
        """
        Create a deep copy of the state.
        """
        new_state = MDPState()
        new_state.token_sequence = self.token_sequence.copy()  # Create a shallow copy of the list
        new_state.step_count = self.step_count
        new_state.stack_size = self.stack_size
        return new_state


class AlphaMiningMDP:
    """
    Full MDP environment for alpha formula construction.

    This environment defines:
    - Episode initialization
    - State transitions via token actions
    - Action validity and domain-specific pruning rules
    """

    def __init__(self):
        self.max_episode_length = 30
        self.current_state = None
        self.formula_evaluator = FormulaEvaluator()  # Unified formula evaluator used throughout the environment

    def reset(self):
        """
        Start a new episode and return the initial state.
        """
        self.current_state = MDPState()
        return self.current_state

    def step(self, action_token):
        """
        Execute one action in the environment.
        """
        if not self.is_valid_action(action_token):
            # Invalid actions immediately terminate the episode with penalty
            return self.current_state, -1.0, True

        self.current_state.add_token(action_token)

        # Episode terminates explicitly at END token
        if action_token == 'END':
            done = True
        else:
            done = False

        # Episode also terminates if maximum length is reached
        if self.current_state.step_count >= self.max_episode_length:
            done = True
            
        # Reward is deferred; immediate reward is zero
        return self.current_state, 0.0, done

    def is_valid_action(self, action_token):
        """
        Check whether an action is syntactically valid under RPN rules.
        """
        valid_actions = RPNValidator.get_valid_next_tokens(self.current_state.token_sequence)
        return action_token in valid_actions

    def get_valid_actions(self, state):
        """
        Retrieve valid actions for the given state, applying additional
        domain-specific constraints to avoid trivial or degenerate formulas.

        This includes:
        - Avoiding constant-only expressions
        - Limiting excessive repetition
        - Enforcing diversity and structural balance
        """
        base_actions = RPNValidator.get_valid_next_tokens(state.token_sequence)


        filtered_actions = []

        for action in base_actions:
 
            if len(state.token_sequence) > 0:
                last_token = state.token_sequence[-1]


                # Prevent time-series operators with too-small windows
                if last_token.name in ['ts_skew'] and action == 'delta_3':
                    continue  

                if last_token.name in ['ts_kurt'] and action == 'delta_3':
                    continue  

                # If stack top is a constant, avoid statistical operators
                # that would still yield constants
                if self.is_stack_top_constant(state):
                    if action in ['ts_std', 'ts_var', 'ts_skew', 'ts_kurt']:
                        continue

            filtered_actions.append(action)

        
        actions = filtered_actions

        # Constraint 1: limit consecutive operands
        # If there are already ≥2 consecutive operands, only allow operators or END
        
        consecutive_operands = 0
        for tok in reversed(state.token_sequence):
            if tok.type == TokenType.OPERAND and tok.name != 'BEG':
                consecutive_operands += 1
            else:
                break
        if consecutive_operands >= 2:
            actions = [a for a in actions
                       if (a == 'END') or (TOKEN_DEFINITIONS[a].type == TokenType.OPERATOR)]

        # Constraint 2: suppress excessive repetition of the same operand
        # If an operand appears ≥3 times in the last 5 steps, temporarily disable it
        recent_names = [t.name for t in state.token_sequence[-5:]]
        counts = {}
        for name in recent_names:
            counts[name] = counts.get(name, 0) + 1
        
        actions = [a for a in actions
                   if not (TOKEN_DEFINITIONS[a].type == TokenType.OPERAND and counts.get(a, 0) >= 3)]
        
        # Fallback: if all actions are filtered out, revert to grammar-level actions
        if not actions:
            actions = base_actions

        if not actions:
            return []

        return actions

    def is_stack_top_constant(self, state):
        """
        Check whether the top-most operand on the implicit RPN stack is a constant.

        This is a heuristic used to prune actions that would only produce constants.
        """
        for token in reversed(state.token_sequence):
            if token.type == TokenType.OPERAND:
                return token.name.startswith('const_')
            if token.type == TokenType.OPERATOR:
                break
        return False

