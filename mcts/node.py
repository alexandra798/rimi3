# mcts/node.py
"""
MCTS node definition

This implementation stores an MDPState at each node, and tracks edge statistics
(N, P, Q, R, W) typically used by PUCT-style selection.
"""

import numpy as np
import math


class MCTSNode:
    """
    Monte Carlo Tree Search node (state-based).

    Each node represents:
    - a state (MDPState)
    - the action taken from the parent to reach this node
    - edge statistics used by PUCT for selection and backup
    """

    def __init__(self, state=None, parent=None, action=None, prior_prob=1.0, c_puct=1.0):
        """
        Initialize an MCTS node.

        Args:
            state: MDPState instance representing the current environment state.
            parent: Parent node.
            action: Action (token name) taken from the parent to reach this node.
            prior_prob: Prior probability P(s,a) provided by a policy network (or heuristic).
            c_puct: Exploration coefficient used in the PUCT formula.
        """
        self.state = state
        self.parent = parent
        self.action = action  # Action taken to reach this node from its parent
        self.c_puct = c_puct

        # Edge statistics (as commonly used in PUCT / AlphaZero-style MCTS)
        self.N = 0          # N(s,a) - visit count
        self.P = prior_prob # P(s,a) - prior probability
        self.Q = 0.0        # Q(s,a) - mean action value
        self.R = 0.0        # R(s,a) - intermediate reward (optional / domain-specific)
        self.W = 0.0        # W(s,a) - cumulative value sum (used to compute Q)

        # Child nodes keyed by action token name: {action: child_node}
        self.children = {}

    def is_expanded(self):
        """Return True if this node has been expanded (i.e., has any children)."""
        return len(self.children) > 0

    def is_terminal(self):
        """
        Return True if this node is terminal.

        Here, a terminal node is defined as one whose state's last token is 'END'.
        """
        if self.state is None:
            return False
        if len(self.state.token_sequence) > 0:
            return self.state.token_sequence[-1].name == 'END'
        return False

    def is_fully_expanded(self):
        """
        Return True if all syntactically valid actions have corresponding child nodes.

        Valid actions are determined by the RPNValidator, based on the current token sequence.
        """
        if self.state is None:
            return False
        from core import RPNValidator
        valid_actions = RPNValidator.get_valid_next_tokens(self.state.token_sequence)
        return all(action in self.children for action in valid_actions)

    def add_child(self, action, child_state, prior_prob=1.0):
        """
        Create and attach a child node for the given action.

        Parameters:
        - action: token name representing the chosen action
        - child_state: resulting MDPState after applying the action
        - prior_prob: prior probability for this edge
        """
        child = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            prior_prob=prior_prob
        )
        self.children[action] = child
        return child

    def update(self, value):
        """
        Update visit count and value estimates using a backed-up evaluation.

        The backup rule used here:
        - N += 1
        - W += value
        - Q = W / N
        """
        # Increment visit count
        self.N += 1

        # Accumulate total value
        self.W += value

        # Mean value estimate
        self.Q = self.W / self.N

    def update_intermediate_reward(self, reward):
        """
        Store intermediate reward R(s,a) for analysis or auxiliary training signals.
        """
        self.R = reward

    # Important
    def get_best_child(self, c_puct=None, diversity_penalty_func=None):
        """
        Select the best child node using the PUCT formula, optionally applying a diversity penalty.

        Args:
            c_puct: Optional override for the exploration coefficient.
            diversity_penalty_func: Optional function mapping a child node -> penalty value
                                   (subtracted from the exploration term).
        """
        if not self.children:
            return None

        c = c_puct if c_puct is not None else self.c_puct

        # Total visits across children; used for exploration scaling
        total_visits = sum(child.N for child in self.children.values())

        # If no child has been visited, prefer the one with the highest valid prior.
        # If priors are unusable, fall back to random choice.
        if total_visits == 0:
            valid_children = [ch for ch in self.children.values()
                              if np.isfinite(ch.P) and ch.P > 0.0]
            if not valid_children:
                import random
                return random.choice(list(self.children.values()))
            return max(valid_children, key=lambda ch: ch.P)

        sqrt_total = math.sqrt(total_visits)
        best_value = -float('inf')
        best_child = None

        for child in self.children.values():
            # Q robustness: treat non-finite Q as 0
            q_value = child.Q
            if not np.isfinite(q_value):
                q_value = 0.0

            # P robustness: if prior is invalid, use a uniform fallback
            p_value = child.P
            if not (np.isfinite(p_value) and p_value > 0.0):
                p_value = 1.0 / len(self.children)

            # Exploration term U: proportional to prior and total visits, inversely to child visits
            u_value = c * p_value * sqrt_total / (1.0 + child.N)

            # Optional diversity penalty (reduces exploration bonus for less diverse children)
            if diversity_penalty_func:
                u_value -= diversity_penalty_func(child)

            # PUCT score
            puct_value = q_value + u_value

            if puct_value > best_value:
                best_value = puct_value
                best_child = child

        return best_child

    def get_visit_distribution(self):
        """
        Return (actions, visits) for child nodes.

        This is typically used to form a policy target (e.g., proportional to visit counts)
        when choosing the final action at the root.
        """
        actions = list(self.children.keys())
        visits = [self.children[a].N for a in actions]
        return actions, visits

    def get_edge_info(self):
        """
        Return edge statistics for debugging/inspection.
        """
        return {
            'N(s,a)': self.N,
            'P(s,a)': self.P,
            'Q(s,a)': self.Q,
            'R(s,a)': self.R
        }

    def __repr__(self):
        """
        Human-readable representation of the node.

        If a state exists, this prints a formula-like token string (excluding the initial BEG).
        """
        if self.state:
            formula = ' '.join([t.name for t in self.state.token_sequence[1:]])
            return f"MCTSNode(formula='{formula}', N={self.N}, Q={self.Q:.4f}, R={self.R:.4f})"
        return f"MCTSNode(root, N={self.N})"
