# alpha/pool.py 完整修复版
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


class AlphaPool:

    def __init__(self, pool_size=100, lambda_param=0.1, learning_rate=0.01,
                 min_std=1e-6, min_unique_ratio=0.01):
        self.pool_size = pool_size
        self.lambda_param = lambda_param
        self.learning_rate = learning_rate

        # 新增：常数检测参数
        self.min_std = min_std  # 最小标准差阈值
        self.min_unique_ratio = min_unique_ratio  # 最小唯一值比例

        self.alphas = []
        self.model = None

        # 新增：统计信息
        self.rejected_constant_count = 0
        self.rejected_low_ic_count = 0

    def is_valid_alpha(self, alpha_values):
        """
        检查alpha是否有效（非常数）

        Args:
            alpha_values: alpha的值序列

        Returns:
            bool: 是否为有效的非常数alpha
        """
        if alpha_values is None:
            return False

        # 转换为numpy数组
        if hasattr(alpha_values, 'values'):
            values = alpha_values.values
        else:
            values = np.array(alpha_values)

        # 检查是否为空
        if len(values) == 0:
            return False

        # 确保是数值类型
        try:
            values = np.array(values, dtype=np.float64)
        except (ValueError, TypeError):
            return False

        # 移除NaN
        valid_values = values[~np.isnan(values)]

        if len(valid_values) < 10:  # 太少有效值
            return False

        # 检查1: 标准差
        std = np.std(valid_values)
        if std < self.min_std:
            logger.debug(f"Alpha rejected: nearly constant (std={std:.8f})")
            self.rejected_constant_count += 1
            return False

        # 检查2: 唯一值比例
        unique_count = len(np.unique(valid_values))
        unique_ratio = unique_count / len(valid_values)
        if unique_ratio < self.min_unique_ratio:
            logger.debug(f"Alpha rejected: too few unique values ({unique_ratio:.1%})")
            self.rejected_constant_count += 1
            return False

        # 检查3: 变异系数（相对标准差）
        mean_val = np.mean(valid_values)
        if abs(mean_val) > 1e-10:  # 避免除零
            cv = std / abs(mean_val)
            if cv < 0.001:  # 变异系数太小
                logger.debug(f"Alpha rejected: low coefficient of variation ({cv:.6f})")
                self.rejected_constant_count += 1
                return False

        return True

    def add_to_pool(self, alpha_info):
        """
        Args: alpha_info: 字典，包含 'formula', 'score', 可选 'values', 'ic'
        """
        # 首先检查是否已存在
        if any(a['formula'] == alpha_info['formula'] for a in self.alphas):
            return

        # 新增：检查alpha有效性
        if 'values' in alpha_info:
            if not self.is_valid_alpha(alpha_info['values']):
                logger.info(f"Rejected constant alpha: {alpha_info['formula'][:50]}...")
                return

        # 新增：检查IC阈值
        if 'ic' in alpha_info and abs(alpha_info.get('ic', 0)) < 0.01:
            logger.debug(f"Rejected low IC alpha: IC={alpha_info['ic']:.4f}")
            self.rejected_low_ic_count += 1
            return

        # 确保有必要的字段
        if 'weight' not in alpha_info:
            alpha_info['weight'] = 1.0 / max(len(self.alphas), 1)
        if 'ic' not in alpha_info and 'score' in alpha_info:
            alpha_info['ic'] = alpha_info['score']

        self.alphas.append(alpha_info)
        logger.info(f"Added valid alpha to pool: {alpha_info['formula'][:50]}... (IC={alpha_info.get('ic', 0):.4f})")

        # 如果超过池大小，移除最差的
        if len(self.alphas) > self.pool_size:
            self._remove_worst_alpha()

    def update_pool(self, X_data, y_data, evaluate_formula):
        """
        更新整个池：评估所有公式并优化权重
        """
        import hashlib

        if isinstance(X_data, pd.DataFrame):
            context_id = hashlib.md5(X_data.index.values.tobytes()).hexdigest()[:8]
        else:
            context_id = "unknown"

        logger.info(f"Updating alpha pool with context {context_id}, {len(self.alphas)} formulas...")

        alphas_to_remove = []

        for i, alpha in enumerate(self.alphas):
            # 每次都重新计算，不依赖缓存的values
            if 'context_id' not in alpha or alpha['context_id'] != context_id:
                try:
                    alpha['values'] = evaluate_formula.evaluate(
                        alpha['formula'],
                        X_data,
                        allow_partial=False
                    )
                    alpha['context_id'] = context_id

                    # 检查有效性
                    if not self.is_valid_alpha(alpha['values']):
                        alphas_to_remove.append(i)
                        continue

                    # 计算IC
                    if alpha['values'] is not None and not alpha['values'].isna().all():
                        alpha['ic'] = self._calculate_ic(alpha['values'], y_data)

                        # 检查IC阈值
                        if abs(alpha['ic']) < 0.01:
                            alphas_to_remove.append(i)
                    else:
                        alphas_to_remove.append(i)

                except Exception as e:
                    logger.warning(f"Failed to evaluate formula: {alpha['formula'][:50]}...")
                    alphas_to_remove.append(i)

        # 2. 移除无效的alpha（这个必须保留！）
        for idx in reversed(alphas_to_remove):
            removed = self.alphas.pop(idx)
            logger.info(f"Removed invalid alpha: {removed['formula'][:50]}...")

        # 3. 优化权重（这个必须保留！需要values）
        if len(self.alphas) > 0:
            self._optimize_weights_gradient_descent(X_data, y_data)

        # 优化完成后，可以选择性地清理values以节省内存
        for alpha in self.alphas:
            if 'values' in alpha:
                del alpha['values']  # 优化完成后删除values

        # 4. 根据IC排序（保留）
        self.alphas.sort(key=lambda x: abs(x.get('ic', 0)), reverse=True)

        # 5. 保持池大小（保留）
        if len(self.alphas) > self.pool_size:
            self.alphas = self.alphas[:self.pool_size]

    def maintain_pool(self, new_alpha, X_data, y_data):
        """
        Algorithm 1: 维护alpha池（保留原有实现）
        输入：当前alpha集合F，新alpha f_new，组合模型c(·|F,ω)
        输出：最优alpha集合F*和权重ω*
        """
        # Step 1: F ← F ∪ f_new
        self.alphas.append(new_alpha)

        # Step 2-4: 梯度下降优化权重
        self._optimize_weights_gradient_descent(X_data, y_data)

        # Step 5-6: 如果超过池大小，移除权重最小的alpha
        if len(self.alphas) > self.pool_size:
            self._remove_worst_alpha()
            # 重新优化权重
            self._optimize_weights_gradient_descent(X_data, y_data)

        return self.alphas

    def get_top_formulas(self, n=5):
        """
        获取最佳的n个公式

        Args:
            n: 返回的公式数量

        Returns:
            公式字符串列表
        """
        # 根据IC或权重排序
        sorted_alphas = sorted(
            self.alphas,
            key=lambda x: abs(x.get('ic', 0) * x.get('weight', 1)),
            reverse=True
        )

        # 返回前n个公式
        top_formulas = []
        for alpha in sorted_alphas[:n]:
            formula = alpha['formula']
            ic = alpha.get('ic', 0)
            weight = alpha.get('weight', 1)
            logger.info(f"Top formula: {formula[:50]}... (IC={ic:.4f}, weight={weight:.4f})")
            top_formulas.append(formula)

        return top_formulas

    def _optimize_weights_gradient_descent(self, X_data, y_data, max_iters=100):
        """使用梯度下降优化权重（论文核心）"""
        if len(self.alphas) == 0:
            return

        # 构建特征矩阵
        feature_matrix = []
        valid_indices = []

        for i, alpha in enumerate(self.alphas):
            if 'values' in alpha and alpha['values'] is not None:
                values = alpha['values']
                if hasattr(values, 'values'):
                    values = values.values
                feature_matrix.append(values.flatten())
                valid_indices.append(i)

        if not feature_matrix:
            logger.warning("No valid alpha values for optimization")
            return

        X = np.column_stack(feature_matrix)
        y = y_data.values if hasattr(y_data, 'values') else y_data

        # 确保维度匹配
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # 移除NaN
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y))
        if valid_mask.sum() < 10:
            logger.warning("Insufficient valid data for optimization")
            return

        X_clean = X[valid_mask]
        y_clean = y[valid_mask]

        # === 新增：列标准化，防止数值爆炸 ===

        with np.errstate(all='ignore'):
            col_mean = X_clean.mean(axis=0)
            col_std = X_clean.std(axis=0)
            eps = 1e-8
            col_std_safe = np.where(col_std < eps, 1.0, col_std)
            X_clean = (X_clean - col_mean) / col_std_safe
            X_clean = np.clip(X_clean, -10, 10)

        # 初始化权重
        weights = np.array([self.alphas[i].get('weight', 1.0 / len(valid_indices))
                            for i in valid_indices])

        # 梯度下降
        best_loss = float('inf')
        best_weights = weights.copy()

        for iteration in range(max_iters):
            # 前向传播：计算预测值
            predictions = X_clean @ weights

            # 计算MSE损失
            error = predictions - y_clean
            loss = np.mean(error ** 2)

            # 保存最佳权重
            if loss < best_loss:
                best_loss = loss
                best_weights = weights.copy()

            # 反向传播：计算梯度
            gradient = 2.0 * (X_clean.T @ error) / len(y_clean)

            # L2正则化
            gradient += self.lambda_param * weights

            # 更新权重
            weights -= self.learning_rate * gradient

            # 早停条件
            if np.linalg.norm(gradient) < 1e-6:
                break

        # 使用最佳权重更新alpha
        for idx, i in enumerate(valid_indices):
            self.alphas[i]['weight'] = best_weights[idx]

        logger.debug(f"Optimized weights after {iteration + 1} iterations, loss={best_loss:.6f}")

    def _remove_worst_alpha(self):
        """移除权重最小的alpha"""
        if len(self.alphas) > 0:
            # 找到绝对权重最小的alpha
            min_weight_idx = np.argmin([abs(a.get('weight', 0)) for a in self.alphas])
            removed_alpha = self.alphas.pop(min_weight_idx)
            logger.info(f"Removed alpha with min weight: {removed_alpha['formula'][:50]}...")

    def _calculate_ic(self, predictions, targets):
        """计算IC（Pearson相关系数）"""
        try:
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(targets, 'values'):
                targets = targets.values

            predictions = np.array(predictions).flatten()
            targets = np.array(targets).flatten()

            # 对齐长度
            min_len = min(len(predictions), len(targets))
            predictions = predictions[:min_len]
            targets = targets[:min_len]

            # 移除NaN
            valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
            if valid_mask.sum() < 2:
                return 0.0

            corr, _ = pearsonr(predictions[valid_mask], targets[valid_mask])
            return corr if not np.isnan(corr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating IC: {e}")
            return 0.0

    def get_composite_alpha_value(self, X_data):
        """计算合成alpha值"""
        if len(self.alphas) == 0:
            return None

        weighted_sum = None
        total_weight = 0

        for alpha in self.alphas:
            if 'values' in alpha and 'weight' in alpha and alpha['values'] is not None:
                values = alpha['values']
                weight = alpha['weight']

                if weighted_sum is None:
                    weighted_sum = weight * values
                else:
                    weighted_sum += weight * values

                total_weight += abs(weight)

        # 归一化
        if weighted_sum is not None and total_weight > 0:
            return weighted_sum / total_weight

        return weighted_sum

    def get_pool_statistics(self):
        """获取池的统计信息"""
        if not self.alphas:
            return {}

        ics = [a.get('ic', 0) for a in self.alphas]
        weights = [a.get('weight', 0) for a in self.alphas]

        # 计算alpha多样性
        if len(self.alphas) > 1:
            correlations = []
            for i in range(len(self.alphas) - 1):
                for j in range(i + 1, len(self.alphas)):
                    if 'values' in self.alphas[i] and 'values' in self.alphas[j]:
                        corr = self._calculate_mutual_ic(
                            self.alphas[i]['values'],
                            self.alphas[j]['values']
                        )
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

            avg_correlation = np.mean(correlations) if correlations else 0
        else:
            avg_correlation = 0

        return {
            'pool_size': len(self.alphas),
            'avg_ic': np.mean(ics),
            'max_ic': np.max(ics),
            'min_ic': np.min(ics),
            'avg_weight': np.mean(weights),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'avg_correlation': avg_correlation,  # 新增：平均相关性
            'rejected_constants': self.rejected_constant_count,  # 新增：拒绝的常数
            'rejected_low_ic': self.rejected_low_ic_count  # 新增：拒绝的低IC
        }