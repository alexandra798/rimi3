"""MCTS节点类定义 - 支持状态而非公式字符串"""
import numpy as np
import math


class MCTSNode:
    """蒙特卡洛树搜索节点 - 基于状态的新版本"""

    def __init__(self, state=None, parent=None, action=None, prior_prob=1.0, c_puct=1.0):
        """
        初始化MCTS节点

        Args:
            state: MDPState对象，表示当前状态
            parent: 母节点
            action: 从母节点到达此节点的动作（Token名称）
            prior_prob: 策略网络给出的先验概率
        """
        self.state = state
        self.parent = parent
        self.action = action  # 到达此节点的动作
        self.c_puct = c_puct

        # 边信息（论文要求）
        self.N = 0  # N(s,a) - 访问次数
        self.P = prior_prob  # P(s,a) - 先验概率
        self.Q = 0.0  # Q(s,a) - 动作价值
        self.R = 0.0  # R(s,a) - 中间奖励
        self.W = 0.0  # 累积奖励（用于计算Q）

        self.children = {}  # {action: child_node}

    def is_expanded(self):
        """检查节点是否已展开"""
        return len(self.children) > 0

    def is_terminal(self):
        """检查是否为终止节点"""
        if self.state is None:
            return False
        if len(self.state.token_sequence) > 0:
            return self.state.token_sequence[-1].name == 'END'
        return False

    def is_fully_expanded(self):
        """检查节点是否已完全展开"""
        if self.state is None:
            return False
        from core import RPNValidator
        valid_actions = RPNValidator.get_valid_next_tokens(self.state.token_sequence)
        return all(action in self.children for action in valid_actions)

    def add_child(self, action, child_state, prior_prob=1.0):
        """添加子节点"""
        child = MCTSNode(
            state=child_state,
            parent=self,
            action=action,
            prior_prob=prior_prob
        )
        self.children[action] = child
        return child

    def update(self, value):
        """更新节点的访问次数和Q值"""
        # 更新访问次数
        self.N += 1

        # 更新累积奖励
        self.W += value

        # Q(s,a) = W(s,a) / N(s,a)
        self.Q = self.W / self.N

    def update_intermediate_reward(self, reward):
        """更新中间奖励R(s,a)"""
        self.R = reward

    # 重要
    def get_best_child(self, c_puct=None, diversity_penalty_func=None):
        """
        使用论文中的PUCT公式选择最佳子节点，可选加入多样性惩罚

        Args:
            c_puct: 探索系数
            diversity_penalty_func: 多样性惩罚函数，接受 child 节点返回惩罚值
        """
        if not self.children:
            return None

        c = c_puct if c_puct is not None else self.c_puct

        # 计算母节点总访问次数
        total_visits = sum(child.N for child in self.children.values())

        # 首次访问，优先选择 P 有效且最大的；否则随机一个
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
            # Q 容错
            q_value = child.Q
            if not np.isfinite(q_value):
                q_value = 0.0

            # P 容错
            p_value = child.P
            if not (np.isfinite(p_value) and p_value > 0.0):
                p_value = 1.0 / len(self.children)

            # 计算 U 值
            u_value = c * p_value * sqrt_total / (1.0 + child.N)

            # 应用多样性惩罚（如果提供）
            if diversity_penalty_func:
                u_value -= diversity_penalty_func(child)

            # PUCT 值
            puct_value = q_value + u_value

            if puct_value > best_value:
                best_value = puct_value
                best_child = child

        return best_child


    def get_visit_distribution(self):
        """获取子节点的访问次数分布（用于最终动作选择）"""
        actions = list(self.children.keys())
        visits = [self.children[a].N for a in actions]
        return actions, visits

    def get_edge_info(self):
        """获取边信息（用于调试）"""
        return {
            'N(s,a)': self.N,
            'P(s,a)': self.P,
            'Q(s,a)': self.Q,
            'R(s,a)': self.R
        }

    def __repr__(self):
        """节点的字符串表示"""
        if self.state:
            formula = ' '.join([t.name for t in self.state.token_sequence[1:]])
            return f"MCTSNode(formula='{formula}', N={self.N}, Q={self.Q:.4f}, R={self.R:.4f})"
        return f"MCTSNode(root, N={self.N})"