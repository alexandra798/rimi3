# mcts/searcher.py
"""
Token-based MCTS searcher.

This module implements an MCTS loop with:
- PUCT-style selection (with optional diversity penalty)
- Expansion using a policy network (priors P(s,a))
- Rollout using the same policy network (optional / simplified)
- Backpropagation with bootstrap-style returns
"""

import numpy as np
import math
import logging
import torch
from core import RPNValidator, TOKEN_TO_INDEX, TOKEN_DEFINITIONS, TokenType


logger = logging.getLogger(__name__)


class MCTSSearcher:
    """
    MCTS searcher implementing PUCT selection and tree search.

    Key additions in this implementation:
    - Diversity-aware priors / penalties to reduce mode collapse in token sequences
    - Optional memory-based bootstrapping using state embeddings from the policy network
    """

    def __init__(self, policy_network=None, device=None, c_puct=1.414, alpha_diversity=0.1):
        self.policy_network = policy_network
        self.device = device

        # Discount factor (paper setup uses gamma=1.0)
        self.gamma = 1.0

        # PUCT exploration coefficient
        self.c_puct = c_puct

        # Diversity strength used to down-weight over-visited subtrees
        self.alpha_diversity = alpha_diversity

        # Subtree frequency counter:
        # key(hash(token sequence)) -> visit-like count, used for diversity shaping
        self.subtree_counter = {}  # key(hash of token seq) -> int

        # Experience memory for approximate value bootstrap:
        # stores tuples (phi_normalized, Qhat)
        self.memory = []  # [(phi_norm, Qhat)]
        self.max_memory = 5000

    def _hash_seq(self, token_seq):
        """Hash a token sequence by token names (stable within a Python process)."""
        return hash(tuple(t.name for t in token_seq))

    def _apply_diversity(self, prior, key):
        """
        Apply diversity shaping to a prior probability.

        Down-weights priors for token sequences that appear frequently in the explored tree:
            prior' = prior * exp(-alpha_diversity * freq)
        """
        f = self.subtree_counter.get(key, 0)
        return float(prior * np.exp(-self.alpha_diversity * f))

    def _embed(self, state):
        """
        Produce a normalized embedding vector for a state using the policy network's hidden output.

        Notes:
        - Requires policy_network(..., return_hidden=True) support.
        - The returned vector is L2-normalized for cosine similarity queries in memory.
        """
        import torch, numpy as np
        enc = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)
        (probs, logp), h = self.policy_network(enc, valid_actions_mask=None, return_log_probs=True, return_hidden=True)
        v = h.squeeze(0).detach().cpu().numpy()
        n = np.linalg.norm(v) + 1e-12
        return v / n

    def _nn_query(self, phi, topk=5, thr=0.95):
        """
        Query the memory by cosine similarity and return a nearest-neighbor Q estimate if confident.

        Parameters:
        - phi: normalized embedding vector for the current state
        - topk: number of nearest candidates to consider
        - thr: similarity threshold; below this, return None
        """
        if not self.memory:
            return None
        import numpy as np
        sims = [(i, float(phi @ m[0])) for i, m in enumerate(self.memory)]
        sims.sort(key=lambda x: x[1], reverse=True)
        sims = [s for s in sims[:topk] if s[1] >= thr]
        if not sims:
            return None
        # Return the nearest neighbor's Q estimate
        return float(self.memory[sims[0][0]][1])

    def _memory_bootstrap(self, node):
        """Try to bootstrap a leaf value from memory using state embeddings; return None on failure."""
        try:
            phi = self._embed(node.state)
            qhat = self._nn_query(phi, topk=3, thr=0.95)
            return qhat
        except Exception:
            return None

    def _diversity_penalty(self, node, beta=0.1):
        """Compute diversity penalty for a node based on how often its token sequence appears."""
        if not hasattr(node, 'state') or not node.state:
            return 0.0

        key = self._hash_seq(node.state.token_sequence)
        freq = self.subtree_counter.get(key, 0)
        # Log penalty avoids over-penalizing high-frequency nodes too aggressively
        return beta * np.log(1 + freq)

    def search_one_iteration(self, root_node, mdp_env, reward_calculator, X_data, y_data):
        # Run a single full MCTS iteration.
        # Phase 1: Selection
        path = []
        current = root_node

        # Use fixed c_puct plus a diversity penalty in PUCT selection
        while current.is_expanded() and not current.is_terminal():
            current = current.get_best_child(
                c_puct=self.c_puct,
                diversity_penalty_func=lambda child: self._diversity_penalty(child, beta=0.1)
            )
            if current is None:
                break
            path.append(current)

            # If the current token sequence is syntactically valid, compute and store intermediate reward R(s,a)
            if RPNValidator.is_valid_partial_expression(current.state.token_sequence):
                intermediate_reward = reward_calculator.calculate_intermediate_reward(
                    current.state, X_data, y_data
                )
                current.update_intermediate_reward(intermediate_reward)


        # Phase 2: Expansion
        leaf_value = 0
        if not current.is_terminal() and current.N >= 0:
            # Expand the leaf node
            leaf_value = self.expand(current, mdp_env)

            # Select one newly expanded child to continue evaluation from
            if current.children:
                # Choose by priors (robustly normalized)
                probs = [child.P for child in current.children.values()]
                probs = np.array(probs, dtype=np.float64)
                probs = np.where(np.isfinite(probs) & (probs >= 0.0), probs, 0.0)
                s = probs.sum()
                if (not np.isfinite(s)) or s <= 0.0:
                    probs = np.full(len(probs), 1.0 / len(probs))
                else:
                    probs = probs / s

                selected_idx = np.random.choice(len(current.children), p=probs)
                selected_action = list(current.children.keys())[selected_idx]
                current = current.children[selected_action]
                path.append(current)

        # Phase 3: Rollout / Evaluation
        if current.is_terminal():
            # Terminal state: compute terminal reward
            value = reward_calculator.calculate_terminal_reward(
                current.state, X_data, y_data
            )
        else:
            # Non-terminal: estimate leaf value via rollout policy
            value = self.rollout(current, mdp_env, reward_calculator, X_data, y_data)

        # Phase 4: Backpropagation
        self.backpropagate(path, value, reward_calculator, X_data, y_data)

        # Extract trajectory for policy network training
        trajectory = self.extract_trajectory(path)

        return trajectory

    def expand(self, node, mdp_env):
        """Expand a node and assign prior probabilities P(s,a) (from policy network or uniform)."""
        if node.state is None:
            return 0

        valid_actions = mdp_env.get_valid_actions(node.state)
        if not valid_actions:
            return 0

        # Get priors from policy network if available; otherwise use uniform distribution
        if self.policy_network:
            action_probs = self.get_policy_predictions(node.state, valid_actions)
        else:
            action_probs = {a: 1.0 / len(valid_actions) for a in valid_actions}

        for action in valid_actions:
            new_state = node.state.copy()
            new_state.add_token(action)
            key = self._hash_seq(new_state.token_sequence)

            raw_prior = action_probs.get(action, 1.0 / len(valid_actions))
            prior = self._apply_diversity(raw_prior, key)

            child = node.add_child(action, new_state, prior_prob=prior)

            # Record frequency for diversity shaping (increment upon creation; could also be done after backup)
            self.subtree_counter[key] = self.subtree_counter.get(key, 0) + 1

            # Memory-guided bootstrap (see below): return a leaf value estimate if available
        qbar = self._memory_bootstrap(node)
        return qbar if qbar is not None else 0.0


    def rollout(self, node, mdp_env, reward_calculator, X_data, y_data, max_depth=30):
        """
        Perform a rollout from a node using the policy network as the rollout policy.

        Returns:
        - v_l: discounted return computed from collected intermediate/terminal rewards
        """
        current_state = node.state.copy()
        cumulative_reward = 0
        depth = 0
        intermediate_rewards = []

        while depth < max_depth and not current_state.token_sequence[-1].name == 'END':
            # Retrieve valid actions for the current rollout state
            valid_actions = mdp_env.get_valid_actions(current_state)
            if not valid_actions:
                break

            # Choose an action from the rollout policy (policy network if present; otherwise random)
            if self.policy_network:
                action_probs = self.get_policy_predictions(current_state, valid_actions)
                probs = [action_probs.get(a, 0.0) for a in valid_actions]
                probs = np.array(probs, dtype=np.float64)
                probs = np.where(np.isfinite(probs) & (probs >= 0.0), probs, 0.0)
                s = probs.sum()
                if (not np.isfinite(s)) or s <= 0.0:
                    probs = np.full(len(valid_actions), 1.0 / len(valid_actions))
                else:
                    probs = probs / s
                action = np.random.choice(valid_actions, p=probs)
            else:
                # Random action selection
                action = np.random.choice(valid_actions)

            # Apply action to state
            current_state.add_token(action)

            # Compute reward for this step
            if action == 'END':
                reward = reward_calculator.calculate_terminal_reward(
                    current_state, X_data, y_data
                )
            else:
                if RPNValidator.is_valid_partial_expression(current_state.token_sequence):
                    reward = reward_calculator.calculate_intermediate_reward(
                        current_state, X_data, y_data
                    )
                else:
                    reward = 0

            intermediate_rewards.append(reward)
            depth += 1

            if action == 'END':
                break

        # Compute discounted return (paper setup: gamma=1)
        v_l = 0
        for reward in reversed(intermediate_rewards):
            v_l = reward + self.gamma * v_l

        return v_l

    # Intelligent rollout is intentionally disabled / not retained


    def backpropagate(self, path, leaf_value, reward_calculator, X_data, y_data):
        """
        Bootstrap-style backpropagation.

        Target return definition:
            G_k = Σ_{i=0}^{l-1-k} γ^i * r_{k+1+i} + γ^{l-k} * v_l

        Where:
        - r are intermediate rewards stored on edges (node.R)
        - v_l is the estimated leaf value (terminal reward or rollout value)
        """
        if not path:
            return
        rewards = []
        for i, node in enumerate(path):
            if i > 0:  # skip root node
                rewards.append(node.R)

        # l is the index of the last node on the path
        l = len(path) - 1

        # Update each node along the path
        for k in range(len(path)):
            G_k = 0
            # Accumulate discounted intermediate rewards from position k
            for i in range(min(l - k, len(rewards) - k)):
                if k + i < len(rewards):
                    G_k += (self.gamma ** i) * rewards[k + i]
            # Add discounted leaf value
            if l >= k:
                G_k += (self.gamma ** (l - k)) * leaf_value
            # Update node statistics (N, W, Q)
            path[k].update(G_k)

        # Store embeddings and value estimates into memory for future bootstrapping
        for nd in path:
            try:
                phi = self._embed(nd.state)
                self.memory.append((phi, float(nd.Q)))
                if len(self.memory) > self.max_memory:
                    self.memory = self.memory[-self.max_memory:]
            except Exception:
                pass

    def extract_trajectory(self, path):
        """
        Extract a trajectory:
            τ = {s_0, a_1, r_1, s_1, ..., s_T}

        Used for training the policy/value networks.
        """
        trajectory = []
        for i, node in enumerate(path):
            if node.parent and node.action:
                trajectory.append((
                    node.parent.state,
                    node.action,
                    node.R  # stored intermediate reward
                ))
        return trajectory

    def get_policy_predictions(self, state, valid_actions):
        """
        Query the policy network for action probabilities P(s,a).

        Returns:
        - probs: dictionary mapping action token -> probability
        """
        if not self.policy_network:
            probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
            return probs, 0

        # Encode state for the policy network
        state_encoding = torch.FloatTensor(state.encode_for_network()).unsqueeze(0).to(self.device)

        # Build a valid-action mask over the full token vocabulary
        valid_actions_mask = torch.zeros(len(TOKEN_TO_INDEX), dtype=torch.bool)
        for action in valid_actions:
            valid_actions_mask[TOKEN_TO_INDEX[action]] = True
        valid_actions_mask = valid_actions_mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.policy_network(state_encoding, valid_actions_mask, return_log_probs=False)

        # Convert to a clean probability dict: non-negative, finite, sums to 1 (otherwise uniform)
        probs = {}
        raw = []
        for action in valid_actions:
            idx = TOKEN_TO_INDEX[action]
            p = float(action_probs[0, idx].item())
            if not (np.isfinite(p) and p >= 0.0):
                p = 0.0
            raw.append(p)

        raw = np.array(raw, dtype=np.float64)
        s = raw.sum()
        if (not np.isfinite(s)) or s <= 0.0:
            raw = np.full_like(raw, 1.0 / len(raw))
        else:
            raw = raw / s

        for action, p in zip(valid_actions, raw):
            probs[action] = float(p)

        return probs

    def get_best_action(self, root_node, temperature=1.0):
        """
        Select the final action from the root based on visit counts.

        Args:
            root_node: root MCTSNode
            temperature: controls stochasticity (0 = greedy argmax)

        Returns:
            action: selected action token name
        """
        if not root_node.children:
            return None

        actions, visits = root_node.get_visit_distribution()

        if temperature == 0:
            # Greedy selection by maximum visit count
            best_idx = np.argmax(visits)
            return actions[best_idx]
        else:
            visits = np.array(visits, dtype=np.float64)
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            visits = np.where(np.isfinite(visits) & (visits >= 0.0), visits, 0.0)
            s = visits.sum()
            if (not np.isfinite(s)) or s <= 0.0:
                probs = np.full(len(actions), 1.0 / len(actions))
            else:
                probs = visits / s
            return np.random.choice(actions, p=probs)
