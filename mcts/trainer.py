# mcts/trainer.py
"""
RiskMiner full trainer

This trainer orchestrates:
- MCTS trajectory collection in an alpha-mining MDP
- Risk-seeking policy optimization (online RL-style updates)
- Optional supervised distillation from root visit distributions (imitation)
- Alpha pool maintenance and periodic statistics reporting
"""

import logging
import numpy as np
import sys
import os
import torch

from alpha.evaluator import FormulaEvaluator
from core import TOKEN_DEFINITIONS
from mcts.reward_calculator import RewardCalculator

from mcts.node import MCTSNode
from mcts.searcher import MCTSSearcher
from mcts.environment import AlphaMiningMDP, MDPState


from policy.network import PolicyNetwork
from policy.optimizer import RiskSeekingOptimizer

logger = logging.getLogger(__name__)


class RiskMinerTrainer:

    def __init__(self, X_data, y_data, device=None, use_sampling=True, sample_size=50000, random_seed=42):
        self.X_data = X_data
        self.y_data = y_data
        self.use_sampling = use_sampling

        # Supervised-learning buffer for distillation:
        # each item is (state_encoding, actions, pi) where pi is the normalized visit distribution at root
        self.sl_buffer = [] # [(state_enc, actions, pi)]

        # Batch size for each supervised distillation step
        self.sl_batch_size = 64  # number of samples per distillation update

        # If the dataset is large, create a fixed subsample for MCTS training
        if use_sampling and len(X_data) > sample_size:
            logger.info(f"Data too large ({len(X_data)} rows), creating sampled version...")
            np.random.seed(random_seed)
            sample_indices = np.random.choice(len(X_data), sample_size, replace=False)
            self.X_train_sample = X_data.iloc[sample_indices]
            self.y_train_sample = y_data.iloc[sample_indices]
            logger.info(f"Using {sample_size} samples for MCTS training (seed={random_seed})")
        else:
            self.X_train_sample = X_data
            self.y_train_sample = y_data

        # Set device (GPU if available, else CPU)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize core components
        self.mdp_env = AlphaMiningMDP()
        self.policy_network = PolicyNetwork().to(self.device)
        self.optimizer = RiskSeekingOptimizer(self.policy_network, device=self.device)
        self.mcts_searcher = MCTSSearcher(
            policy_network=self.policy_network,
            device=self.device,
            c_puct=1.414  # fixed c_puct
        )
        self.alpha_pool = []
        self.reward_calculator = RewardCalculator(self.alpha_pool, random_seed=random_seed)
        self.formula_evaluator = FormulaEvaluator()  # unified evaluator instance

        logger.info(f"Policy network moved to {self.device}")




    def train(self, num_iterations=200, num_simulations_per_iteration=50):
        """Main training loop."""
        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # Fix the sample for this iteration (stabilizes reward + caching behavior)
            self.reward_calculator.set_iteration_sample(
                self.X_train_sample, self.y_train_sample
            )

            # Periodically clear caches to reduce memory usage / stale entries
            if iteration > 0 and iteration % 10 == 0:
                self.formula_evaluator.clear_cache()
                self.reward_calculator._cache.clear()
                logger.info("Cleared caches to free memory")

            # Phase 1: collect trajectories via MCTS
            trajectories = self.collect_trajectories_with_mcts(
                num_episodes=10,
                num_simulations_per_episode=num_simulations_per_iteration
            )

            # Phase 2: train policy network using collected trajectories
            if self.optimizer and trajectories:
                avg_loss = self.train_policy_network(trajectories)
                logger.info(f"Policy network loss: {avg_loss:.4f}")

            # After phase 2: additional supervised distillation from root visit distribution
            if hasattr(self, 'sl_buffer') and len(self.sl_buffer) >= self.sl_batch_size:
                sl_batch = self.sl_buffer[:self.sl_batch_size]
                del self.sl_buffer[:self.sl_batch_size]
                if hasattr(self.optimizer, 'supervised_update'):
                    sl_loss = self.optimizer.supervised_update(sl_batch)
                    logger.info(f"Supervised distillation loss: {sl_loss:.4f}")

            # Phase 3: evaluate trajectories and update alpha pool
            self.update_alpha_pool(trajectories, iteration)

            # Print periodic statistics
            if (iteration + 1) % 10 == 0:
                self.print_statistics()




    def search_one_iteration(self, root):
        """Run a single MCTS search iteration from the given root node."""
        return self.mcts_searcher.search_one_iteration(
            root_node=root,
            mdp_env=self.mdp_env,
            reward_calculator=self.reward_calculator,
            X_data=self.X_train_sample,  # use sampled dataset (not full) for speed
            y_data=self.y_train_sample  # use sampled dataset (not full) for speed
        )

    def collect_trajectories_with_mcts(self, num_episodes, num_simulations_per_episode):
        """Collect training trajectories using MCTS."""
        all_trajectories = []

        for episode in range(num_episodes):
            logger.debug(f"Starting episode {episode + 1}/{num_episodes}")
            initial_state = self.mdp_env.reset()
            root = MCTSNode(state=initial_state)

            # Run MCTS simulations for this episode
            for sim in range(num_simulations_per_episode):
                if sim % 10 == 0:
                    logger.debug(f"  Simulation {sim}/{num_simulations_per_episode}")
                trajectory = self.search_one_iteration(root)

            # Extract root visit distribution for supervised imitation learning (distillation)
            if root.children:
                actions, visits = root.get_visit_distribution()
                if actions and visits:
                    import numpy as np
                    pi = np.asarray(visits, dtype=float)
                    if np.isfinite(pi).sum() > 0:
                        pi = pi / (pi.sum() + 1e-12)
                        state_enc = root.state.encode_for_network()
                        self.sl_buffer.append((state_enc, actions, pi))

            # Extract a best trajectory from the built search tree
            final_trajectory = self.extract_best_trajectory(root)
            if final_trajectory:
                all_trajectories.append(final_trajectory)
                formula_str = self.get_formula_from_trajectory(final_trajectory)
                if formula_str:
                    logger.debug(f"Episode {episode + 1}: {formula_str}")

        return all_trajectories

    def train_policy_network(self, trajectories):
        """Train the policy network on collected episode trajectories."""
        if not self.optimizer:
            return 0.0

        total_loss = 0.0
        num_updates = 0

        for trajectory in trajectories:
            if not trajectory:
                continue
            updated, loss = self.optimizer.train_on_episode(trajectory)
            if updated:
                total_loss += abs(loss)
                num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)
        if num_updates > 0:
            logger.info(f"Policy network updated: {num_updates}/{len(trajectories)} episodes")
            logger.info(f"Current quantile: {self.optimizer.quantile_estimate:.4f}")
        return avg_loss

    def extract_best_trajectory(self, root):
        """
        Extract a "best" trajectory from an MCTS tree.

        Selection rule:
        - At each step, choose the child with the highest visit count N.

        Additional safety:
        - Verify actions are still valid under current state's RPN constraints before appending.
        - Try to complete the formula with required delta_* parameters and END when possible.
        """
        trajectory = []
        current = root
        max_depth = 30
        depth = 0
        from core import RPNValidator

        while current.children and not current.is_terminal() and depth < max_depth:
            # Choose the most-visited child
            best_action = None
            best_visits = -1
            for action, child in current.children.items():
                if child.N > best_visits:
                    best_visits = child.N
                    best_action = action
                    best_child = child

            if best_action is None:
                break

            # Compute reward for this step (terminal vs intermediate)
            if best_child.state.token_sequence[-1].name == 'END':
                reward = self.reward_calculator.calculate_terminal_reward(
                    best_child.state, self.X_train_sample, self.y_train_sample
                )

            else:
                reward = self.reward_calculator.calculate_intermediate_reward(
                    best_child.state, self.X_train_sample, self.y_train_sample
                )

            valid_now = RPNValidator.get_valid_next_tokens(current.state.token_sequence)
            if best_action not in valid_now:
                # Illegal action detected: stop building the trajectory
                logger.debug(f"Illegal action {best_action} detected, stopping trajectory")
                break

            trajectory.append((current.state, best_action, reward))
            current = best_child
            depth += 1

        # If not terminal yet, attempt to complete the expression when possible
        if not current.is_terminal() and depth < max_depth:
            valid_actions = RPNValidator.get_valid_next_tokens(current.state.token_sequence)

            # If only delta_* tokens are valid next actions, add an appropriate delta to satisfy min window constraints
            if valid_actions and all(a.startswith('delta_') for a in valid_actions):
                last_token = current.state.token_sequence[-1]
                min_window = TOKEN_DEFINITIONS[last_token.name].min_window or 3

                suitable_delta = None
                for delta in valid_actions:
                    delta_value = TOKEN_DEFINITIONS[delta].value
                    if delta_value >= min_window:
                        suitable_delta = delta
                        break

                if suitable_delta:
                    # Add delta
                    delta_state = current.state.copy()
                    delta_state.add_token(suitable_delta)
                    delta_reward = self.reward_calculator.calculate_intermediate_reward(
                        delta_state, self.X_train_sample, self.y_train_sample
                    )
                    trajectory.append((current.state, suitable_delta, delta_reward))
                    # Keep current node type unchanged; use delta_state as a new state for termination checks

                    # Attempt to add END using delta_state
                    if RPNValidator.can_terminate(delta_state.token_sequence):
                        terminal_state = delta_state.copy()
                        terminal_state.add_token('END')
                        terminal_reward = self.reward_calculator.calculate_terminal_reward(
                            terminal_state, self.X_train_sample, self.y_train_sample
                        )
                        trajectory.append((delta_state, 'END', terminal_reward))
                else:
                    # If no suitable delta exists, attempt to terminate directly from current.state
                    if RPNValidator.can_terminate(current.state.token_sequence):
                        terminal_state = current.state.copy()
                        terminal_state.add_token('END')
                        terminal_reward = self.reward_calculator.calculate_terminal_reward(
                            terminal_state, self.X_train_sample, self.y_train_sample
                        )
                        trajectory.append((current.state, 'END', terminal_reward))

        return trajectory

    def get_formula_from_trajectory(self, trajectory):
        """Convert a trajectory into its token sequence (formula representation)."""
        if not trajectory:
            return None

        # Rebuild a complete state by replaying actions
        state = MDPState()
        for s, action, r in trajectory:
            state.add_token(action)

        # Return a readable representation (currently returns the token list, not a joined string)
        return state.token_sequence

    def update_alpha_pool(self, trajectories, iteration):
        """
        Update alpha pool.

        Cold-start policy:
        - For the first few iterations, use a lower IC threshold so the pool can grow.
        """
        new_formulas = []
        constant_count = 0

        X_sample = self.X_train_sample
        y_sample = self.y_train_sample

        for trajectory in trajectories:
            if not trajectory:
                continue

            # Rebuild the final state by replaying actions
            final_state = MDPState()
            for state, action, reward in trajectory:
                final_state.add_token(action)

            # Only consider properly terminated formulas
            if final_state.token_sequence[-1].name == 'END':
                formula_rpn = ' '.join([t.name for t in final_state.token_sequence])
                alpha_values = self.formula_evaluator.evaluate(formula_rpn, X_sample)

                if alpha_values is not None and not alpha_values.isna().all():
                    # Filter constant-like formulas
                    valid_values = alpha_values.dropna()
                    if len(valid_values) > 10:
                        std = valid_values.std()
                        if std < 1e-6:
                            constant_count += 1
                            logger.debug(f"Skipping constant formula")
                            continue

                    # Compute IC for the candidate alpha
                    ic = self.reward_calculator.calculate_ic(alpha_values, y_sample)

                    # Cold-start: relaxed threshold for first 3 iterations
                    if iteration < 3:
                        min_ic_threshold = 0.005
                    else:
                        min_ic_threshold = 0.01

                    if abs(ic) >= min_ic_threshold:
                        new_formulas.append({
                            'formula': formula_rpn,
                            'ic': ic,
                            'values': alpha_values,
                            'iteration': iteration
                        })

        # Add new formulas to the pool (deduplicated by formula string)
        for formula_info in new_formulas:
            exists = any(a['formula'] == formula_info['formula'] for a in self.alpha_pool)
            if not exists:
                self.alpha_pool.append(formula_info)
                logger.info(f"New formula added: IC={formula_info['ic']:.4f}")

        # Maintain pool size
        if len(self.alpha_pool) > 100:
            self.alpha_pool.sort(key=lambda x: abs(x['ic']), reverse=True)
            self.alpha_pool = self.alpha_pool[:100]

        if constant_count > 0:
            logger.info(f"Filtered {constant_count} constant formulas")


    def print_statistics(self):
        """Print enhanced training statistics."""
        logger.info("\n" + "=" * 60)
        logger.info("Training Statistics")
        logger.info("=" * 60)

        # Alpha pool stats
        logger.info(f"Alpha pool size: {len(self.alpha_pool)}")

        if self.alpha_pool:
            # IC distribution
            ics = [a.get('ic', 0) for a in self.alpha_pool]
            logger.info(f"IC distribution: mean={np.mean(ics):.4f}, std={np.std(ics):.4f}")
            logger.info(f"IC range: [{np.min(ics):.4f}, {np.max(ics):.4f}]")

            # Top 5 alphas by absolute IC
            top_5 = sorted(self.alpha_pool, key=lambda x: abs(x['ic']), reverse=True)[:5]
            logger.info("\nTop 5 Alphas by |IC|:")
            for i, alpha in enumerate(top_5, 1):
                formula = alpha['formula']
                if len(formula) > 60:
                    formula = formula[:57] + "..."
                logger.info(f"  {i}. IC={alpha['ic']:+.4f} | {formula}")

        # Cache stats (if evaluator exposes hit/miss counters)
        if hasattr(self.formula_evaluator, '_cache_hits'):
            total_calls = self.formula_evaluator._cache_hits + self.formula_evaluator._cache_misses
            if total_calls > 0:
                hit_rate = self.formula_evaluator._cache_hits / total_calls * 100
                logger.info(f"\nCache hit rate: {hit_rate:.1f}% ({self.formula_evaluator._cache_hits}/{total_calls})")

        # Constant-filter diagnostics
        if hasattr(self.reward_calculator, 'constant_penalty_count'):
            logger.info(f"Constants filtered: {self.reward_calculator.constant_penalty_count}")

        # Quantile estimate from risk-seeking optimizer
        if self.optimizer:
            logger.info(f"Quantile estimate: {self.optimizer.quantile_estimate:.4f}")

        # Diversity stats from subtree counter
        if hasattr(self.mcts_searcher, 'subtree_counter'):
            unique_subtrees = len(self.mcts_searcher.subtree_counter)
            max_freq = max(self.mcts_searcher.subtree_counter.values()) if self.mcts_searcher.subtree_counter else 0
            logger.info(f"Unique subtrees explored: {unique_subtrees}, max frequency: {max_freq}")

        logger.info("=" * 60 + "\n")

    def get_top_formulas(self, n=5):
        """Return the top-n formulas ranked by IC (descending)."""
        if not self.alpha_pool:
            return []

        sorted_pool = sorted(self.alpha_pool, key=lambda x: x['ic'], reverse=True)
        return [alpha['formula'] for alpha in sorted_pool[:n]]
