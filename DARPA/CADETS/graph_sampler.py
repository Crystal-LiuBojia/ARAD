import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import os
import pickle
from typing import List, Tuple, Dict
import torch.nn.functional as F
import resource
import atexit
from model import *


class OptimizedGraphSampler:
    def __init__(self, memory, neighbor_loader, eta=0.5, epsilon=2, k=2, device='cuda',
                 cache_dir='./sampling_cache', num_workers=4):
        self.memory = memory
        self.neighbor_loader = neighbor_loader
        self.eta = eta
        self.epsilon = epsilon
        self.k = k
        self.device = device
        self.cache_dir = cache_dir
        self.num_workers = min(num_workers, mp.cpu_count())

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 缓存数据结构
        self.temporal_positive_cache = {}
        self.temporal_negative_cache = {}
        self.structural_positive_cache = {}
        self.structural_negative_cache = {}

        # 新增：内存状态缓存
        self.memory_state_cache = {}
        self.memory_cache_size = 10000  # 缓存大小限制
        self.memory_cache_hits = 0
        self.memory_cache_misses = 0

        # 预计算的时间窗口
        self.time_windows = self._precompute_time_windows()

        # 设置文件描述符限制
        self._set_file_descriptor_limit()

    def _set_file_descriptor_limit(self):
        """设置文件描述符限制"""
        try:
            # 获取当前限制
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            # 设置新的限制（如果可能）
            new_soft = min(hard, 65535)  # 使用一个合理的较大值
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        except Exception as e:
            print(f"Warning: Could not set file descriptor limit: {e}")

    def _precompute_time_windows(self) -> Dict:
        """预计算时间窗口，用于加速时序采样"""
        windows = {}
        for t in range(0, 1000, 100):  # 可以根据实际时间范围调整
            windows[t] = t + self.eta
        return windows

    def _cache_key(self, node: int, timestamp: float, sampling_type: str) -> str:
        """生成缓存键"""
        return f"{sampling_type}_{node}_{timestamp}"

    def _load_cache(self, cache_type: str) -> Dict:
        """加载缓存"""
        cache_path = os.path.join(self.cache_dir, f"{cache_type}_cache.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def _save_cache(self, cache: Dict, cache_type: str):
        """保存缓存"""
        cache_path = os.path.join(self.cache_dir, f"{cache_type}_cache.pkl")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)

    def _check_memory_updates(self):
        """检查内存更新状态，决定是否需要重新预计算"""
        current_updates = self.memory.get_update_count()  # 需要在memory类中实现这个方法
        if current_updates - self.last_memory_update > self.memory_update_threshold:
            self._invalidate_cache()
            self.last_memory_update = current_updates
            return True
        return False

    def _invalidate_cache(self):
        """使缓存失效"""
        self.temporal_cache.clear()
        self.structural_cache.clear()
        self.edge_index_cache.clear()

    def _get_memory_state(self, node_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取节点内存状态，使用缓存机制"""
        # 将节点ID转换为列表以便于缓存查找
        node_list = node_ids.tolist()

        # 检查缓存中是否已有这些节点的状态
        cached_states = []
        uncached_nodes = []
        uncached_indices = []

        for i, node in enumerate(node_list):
            if node in self.memory_state_cache:
                cached_states.append(self.memory_state_cache[node])
                self.memory_cache_hits += 1
            else:
                uncached_nodes.append(node)
                uncached_indices.append(i)
                self.memory_cache_misses += 1

        # 如果有未缓存的节点，批量获取它们的状态
        if uncached_nodes:
            uncached_tensor = torch.tensor(uncached_nodes, device=self.device)
            new_states = self.memory(uncached_tensor)

            # 更新缓存
            for node, state in zip(uncached_nodes, zip(*new_states)):
                if len(self.memory_state_cache) >= self.memory_cache_size:
                    # 如果缓存已满，随机删除一个条目
                    self.memory_state_cache.pop(next(iter(self.memory_state_cache)))
                self.memory_state_cache[node] = state

        # 合并缓存和新的状态
        all_states = []
        cache_idx = 0
        new_idx = 0

        for i in range(len(node_list)):
            if i in uncached_indices:
                if new_idx < len(uncached_nodes):
                    all_states.append((new_states[0][new_idx], new_states[1][new_idx]))
                    new_idx += 1
                else:
                    # 如果获取新状态失败，使用零张量
                    memory_dim = self.memory.memory_dim
                    all_states.append((torch.zeros(memory_dim, device=self.device),
                                       torch.zeros(1, device=self.device)))
            else:
                if cache_idx < len(cached_states):
                    all_states.append(cached_states[cache_idx])
                    cache_idx += 1
                else:
                    # 如果缓存状态获取失败，使用零张量
                    memory_dim = self.memory.memory_dim
                    all_states.append((torch.zeros(memory_dim, device=self.device),
                                       torch.zeros(1, device=self.device)))

        # 将状态转换为张量
        try:
            z = torch.stack([state[0] for state in all_states])
            last_update = torch.stack([state[1] for state in all_states])
        except Exception as e:
            print(f"Error stacking states: {e}")
            # 如果堆叠失败，返回零张量
            memory_dim = self.memory.memory_dim
            z = torch.zeros((len(node_list), memory_dim), device=self.device)
            last_update = torch.zeros((len(node_list), 1), device=self.device)

        return z, last_update

    def _eta_bfs_sampling_worker(self, node: int, timestamp: float, is_positive: bool = True) -> List[int]:
        """时序采样的工作函数 - 返回采样节点的索引列表

        Args:
            node: 中心节点
            timestamp: 当前时间戳
            is_positive: 是否为正样本采样（True为正样本，False为负样本）
        """
        # print("eta_bfs_sampling_worker")
        cache_key = self._cache_key(node, timestamp, 'temporal_positive' if is_positive else 'temporal_negative')
        cache = self.temporal_positive_cache if is_positive else self.temporal_negative_cache

        if cache_key in cache:
            # print("temporal cache")
            return cache[cache_key]

        # 获取邻居
        n_id, edge_index, e_id = self.neighbor_loader(torch.tensor([node], device=self.device))

        if len(n_id) <= 1:  # 没有邻居
            return []

        # 获取内存状态
        z, last_update = self.memory(n_id)

        if len(last_update) == 0 or e_id is None or len(e_id) == 0:
            return []

        if (edge_index[0] >= len(last_update)).any():
            return []

        # 计算时间间隔
        rel_t = last_update[edge_index[0]] - timestamp

        if len(rel_t) == 0 or torch.all(rel_t == 0):
            return []

        # 计算归一化时间间隔
        min_t = rel_t.min()
        max_t = rel_t.max()
        normalized_times = (rel_t - min_t) / (max_t - min_t + 1e-6)

        # 根据是否为正样本选择不同的时间计算方式
        if not is_positive:
            normalized_times = 1 - normalized_times

        # 计算采样概率
        probs = F.softmax(normalized_times / self.temperature, dim=0)

        # 采样邻居
        num_samples = min(len(n_id) - 1, max(1, int(self.eta * (len(n_id) - 1))))
        num_samples = min(num_samples, len(probs))

        if num_samples <= 0:
            return []

        try:
            sampled_indices = torch.multinomial(probs, num_samples)
            valid_indices = sampled_indices[sampled_indices < len(n_id) - 1]
            if len(valid_indices) == 0:
                return []

            sampled_neighbors = n_id[valid_indices + 1].tolist()

            # 保存到缓存
            cache[cache_key] = sampled_neighbors

        except Exception as e:
            print(f"Error sampling neighbors: {e}")
            return []

        return sampled_neighbors

    def _epsilon_dfs_sampling_worker(self, node: int, timestamp: float, is_positive: bool = True) -> List[int]:
        """结构采样的工作函数 - 返回采样节点的索引列表

        Args:
            node: 中心节点
            timestamp: 当前时间戳
            is_positive: 是否为正样本采样（True为正样本，False为负样本）
        """
        # print("epsilon_dfs_sampling_worker")
        cache_key = self._cache_key(node, timestamp, 'structural_positive' if is_positive else 'structural_negative')
        cache = self.structural_positive_cache if is_positive else self.structural_negative_cache

        if cache_key in cache:
            # print("structural cache")
            return cache[cache_key]

        # 使用DFS进行结构采样
        visited = set()
        stack = [(node, 0)]  # (node, depth)
        sampled_nodes = []

        while stack and len(sampled_nodes) < self.k:
            current_node, depth = stack.pop()
            if current_node not in visited and depth <= self.epsilon:
                visited.add(current_node)
                sampled_nodes.append(current_node)

                # 获取当前节点的邻居
                neighbors = self._get_neighbors_sparse(current_node)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))

        # 保存到缓存
        cache[cache_key] = sampled_nodes
        return sampled_nodes

    def precompute_sampling(self, nodes: List[int], timestamps: List[float]):
        """预计算并缓存采样节点索引，包括正负样本"""
        print("Checking for existing cache files...")

        # 尝试加载时序采样缓存（正样本）
        temporal_pos_cache = self._load_cache('temporal_positive')
        if temporal_pos_cache:
            print("Found existing temporal positive cache, loading...")
            self.temporal_positive_cache = temporal_pos_cache
        else:
            print("No temporal positive cache found, computing temporal positive sampling results...")
            # 并行处理时序正样本采样
            temporal_pos_results = self._parallel_temporal_sampling(nodes, timestamps, is_positive=True)
            for node, ts, result in zip(nodes, timestamps, temporal_pos_results):
                self.temporal_positive_cache[self._cache_key(node, ts, 'temporal_positive')] = result
            # 保存时序正样本采样缓存
            self._save_cache(self.temporal_positive_cache, 'temporal_positive')

        # 尝试加载时序采样缓存（负样本）
        temporal_neg_cache = self._load_cache('temporal_negative')
        if temporal_neg_cache:
            print("Found existing temporal negative cache, loading...")
            self.temporal_negative_cache = temporal_neg_cache
        else:
            print("No temporal negative cache found, computing temporal negative sampling results...")
            # 并行处理时序负样本采样
            temporal_neg_results = self._parallel_temporal_sampling(nodes, timestamps, is_positive=False)
            for node, ts, result in zip(nodes, timestamps, temporal_neg_results):
                self.temporal_negative_cache[self._cache_key(node, ts, 'temporal_negative')] = result
            # 保存时序负样本采样缓存
            self._save_cache(self.temporal_negative_cache, 'temporal_negative')

        # 尝试加载结构采样缓存（正样本）
        structural_pos_cache = self._load_cache('structural_positive')
        if structural_pos_cache:
            print("Found existing structural positive cache, loading...")
            self.structural_positive_cache = structural_pos_cache
        else:
            print("No structural positive cache found, computing structural positive sampling results...")
            # 并行处理结构正样本采样
            structural_pos_results = self._parallel_structural_sampling(nodes, timestamps, is_positive=True)
            for node, ts, result in zip(nodes, timestamps, structural_pos_results):
                self.structural_positive_cache[self._cache_key(node, ts, 'structural_positive')] = result
            # 保存结构正样本采样缓存
            self._save_cache(self.structural_positive_cache, 'structural_positive')

        # 尝试加载结构采样缓存（负样本）
        structural_neg_cache = self._load_cache('structural_negative')
        if structural_neg_cache:
            print("Found existing structural negative cache, loading...")
            self.structural_negative_cache = structural_neg_cache
        else:
            print("No structural negative cache found, computing structural negative sampling results...")
            # 并行处理结构负样本采样
            structural_neg_results = self._parallel_structural_sampling(nodes, timestamps, is_positive=False)
            for node, ts, result in zip(nodes, timestamps, structural_neg_results):
                self.structural_negative_cache[self._cache_key(node, ts, 'structural_negative')] = result
            # 保存结构负样本采样缓存
            self._save_cache(self.structural_negative_cache, 'structural_negative')

    def _parallel_temporal_sampling(self, nodes: List[int], timestamps: List[float], is_positive: bool = True) -> List[
        List[int]]:
        """并行处理时序采样

        Args:
            nodes: 节点列表
            timestamps: 时间戳列表
            is_positive: 是否为正样本采样
        """
        # 使用上下文管理器确保资源正确释放
        with mp.Pool(processes=self.num_workers, maxtasksperchild=100) as pool:
            try:
                results = pool.starmap(
                    partial(self._eta_bfs_sampling_worker, is_positive=is_positive),
                    zip(nodes, timestamps)
                )
            except Exception as e:
                print(f"Error in temporal sampling: {e}")
                # 如果出错，返回空列表列表
                results = [[] for _ in range(len(nodes))]
        return results

    def _parallel_structural_sampling(self, nodes: List[int], timestamps: List[float], is_positive: bool = True) -> \
            List[List[int]]:
        """并行处理结构采样

        Args:
            nodes: 节点列表
            timestamps: 时间戳列表
            is_positive: 是否为正样本采样
        """
        # 使用上下文管理器确保资源正确释放
        with mp.Pool(processes=self.num_workers, maxtasksperchild=100) as pool:
            try:
                results = pool.starmap(
                    partial(self._epsilon_dfs_sampling_worker, is_positive=is_positive),
                    zip(nodes, timestamps)
                )
            except Exception as e:
                print(f"Error in structural sampling: {e}")
                # 如果出错，返回空列表列表
                results = [[] for _ in range(len(nodes))]
        return results

    def _get_neighbors_sparse(self, node: int, edge_mask: torch.Tensor = None) -> torch.Tensor:
        """使用稀疏矩阵加速邻居查找"""
        # 获取节点的邻居
        n_id, edge_index, e_id = self.neighbor_loader(torch.tensor([node], device=self.device))

        if len(n_id) <= 1:  # 没有邻居
            return torch.empty(0, dtype=torch.long, device=self.device)

        # 获取内存状态
        z, last_update = self.memory(n_id)

        # 如果没有边索引，返回空张量
        if edge_index is None or len(edge_index) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        # 获取源节点和目标节点
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # 找到当前节点的邻居
        src_mask = src_nodes == 0  # 0是当前节点在n_id中的索引
        dst_mask = dst_nodes == 0

        # 获取邻居节点
        src_neighbors = dst_nodes[src_mask]
        dst_neighbors = src_nodes[dst_mask]

        # 合并邻居并去重
        neighbors = torch.unique(torch.cat([src_neighbors, dst_neighbors]))

        # 将邻居索引映射回原始节点ID
        if len(neighbors) > 0:
            neighbors = n_id[neighbors]

        return neighbors

    def batch_sample(self, nodes: torch.Tensor, timestamps: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量采样接口 - 使用缓存的节点索引实时计算嵌入"""
        # 将输入转换为列表以便于处理
        nodes_list = nodes.tolist()
        timestamps_list = timestamps.tolist()

        # 收集所有需要获取内存状态的节点
        all_nodes = set()
        for node, ts in zip(nodes_list, timestamps_list):
            # 获取时序采样的节点索引
            pos_nodes = self._eta_bfs_sampling_worker(node, ts, is_positive=True)
            neg_nodes = self._eta_bfs_sampling_worker(node, ts, is_positive=False)

            # 获取结构采样的节点索引
            struct_pos_nodes = self._epsilon_dfs_sampling_worker(node, ts, is_positive=True)
            struct_neg_nodes = self._epsilon_dfs_sampling_worker(node, ts, is_positive=False)

            # 添加到集合中
            all_nodes.update([node])
            all_nodes.update(pos_nodes)
            all_nodes.update(neg_nodes)
            all_nodes.update(struct_pos_nodes)
            all_nodes.update(struct_neg_nodes)

        # 批量获取所有节点的内存状态
        all_nodes_tensor = torch.tensor(list(all_nodes), device=self.device)
        all_states = self._get_memory_state(all_nodes_tensor)

        # 创建节点ID到索引的映射
        node_to_idx = {node.item(): idx for idx, node in enumerate(all_nodes_tensor)}

        # 初始化结果张量
        memory_dim = self.memory.memory_dim
        batch_size = len(nodes_list)
        temporal_pos_tensor = torch.zeros((batch_size, memory_dim), device=self.device)
        temporal_neg_tensor = torch.zeros((batch_size, memory_dim), device=self.device)
        structural_pos_tensor = torch.zeros((batch_size, memory_dim), device=self.device)
        structural_neg_tensor = torch.zeros((batch_size, memory_dim), device=self.device)

        # 处理每个样本
        for i, (node, ts) in enumerate(zip(nodes_list, timestamps_list)):
            try:
                # 获取时序采样的节点索引
                pos_nodes = self._eta_bfs_sampling_worker(node, ts, is_positive=True)
                neg_nodes = self._eta_bfs_sampling_worker(node, ts, is_positive=False)

                # 获取结构采样的节点索引
                struct_pos_nodes = self._epsilon_dfs_sampling_worker(node, ts, is_positive=True)
                struct_neg_nodes = self._epsilon_dfs_sampling_worker(node, ts, is_positive=False)

                # 从缓存中获取嵌入
                if pos_nodes:
                    pos_indices = [node_to_idx[n] for n in pos_nodes if n in node_to_idx]
                    if pos_indices:
                        temporal_pos_tensor[i] = all_states[0][pos_indices].mean(dim=0)
                    else:
                        temporal_pos_tensor[i] = torch.ones(memory_dim, device=self.device)
                else:
                    temporal_pos_tensor[i] = torch.ones(memory_dim, device=self.device)

                if neg_nodes:
                    neg_indices = [node_to_idx[n] for n in neg_nodes if n in node_to_idx]
                    if neg_indices:
                        temporal_neg_tensor[i] = all_states[0][neg_indices].mean(dim=0)
                    else:
                        temporal_neg_tensor[i] = torch.ones(memory_dim, device=self.device)
                else:
                    temporal_neg_tensor[i] = torch.ones(memory_dim, device=self.device)

                if struct_pos_nodes:
                    struct_pos_indices = [node_to_idx[n] for n in struct_pos_nodes if n in node_to_idx]
                    if struct_pos_indices:
                        structural_pos_tensor[i] = all_states[0][struct_pos_indices].mean(dim=0)
                    else:
                        structural_pos_tensor[i] = torch.ones(memory_dim, device=self.device)
                else:
                    structural_pos_tensor[i] = torch.ones(memory_dim, device=self.device)

                if struct_neg_nodes:
                    struct_neg_indices = [node_to_idx[n] for n in struct_neg_nodes if n in node_to_idx]
                    if struct_neg_indices:
                        structural_neg_tensor[i] = all_states[0][struct_neg_indices].mean(dim=0)
                    else:
                        structural_neg_tensor[i] = torch.ones(memory_dim, device=self.device)
                else:
                    structural_neg_tensor[i] = torch.ones(memory_dim, device=self.device)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # 如果处理失败，使用单位向量
                temporal_pos_tensor[i] = torch.ones(memory_dim, device=self.device)
                temporal_neg_tensor[i] = torch.ones(memory_dim, device=self.device)
                structural_pos_tensor[i] = torch.ones(memory_dim, device=self.device)
                structural_neg_tensor[i] = torch.ones(memory_dim, device=self.device)

        return (temporal_pos_tensor, temporal_neg_tensor,
                structural_pos_tensor, structural_neg_tensor)


class DualBranchContrastiveLearning:
    def __init__(self,
                 temperature: float = 0.1,
                 beta: float = 0.5,
                 distance_metric: str = 'cosine'):
        self.temperature = temperature
        self.beta = beta
        self.distance_metric = distance_metric

    def _compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算两个张量之间的相似度"""
        # 检查输入张量是否为空
        if x.numel() == 0 or y.numel() == 0:
            return torch.tensor(0.0, device=x.device)

        # 确保输入张量的维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        # 确保张量维度匹配
        if x.size(-1) != y.size(-1):
            raise ValueError(f"Feature dimensions do not match: {x.size(-1)} vs {y.size(-1)}")

        if self.distance_metric == 'cosine':
            return F.cosine_similarity(x, y, dim=-1)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.distance_metric}")

    def temporal_contrast(self,
                          node_embedding: torch.Tensor,
                          positive_subgraph: torch.Tensor,
                          negative_subgraph: torch.Tensor) -> torch.Tensor:
        """计算时序InfoNCE损失"""
        # 检查输入是否为空
        if node_embedding.numel() == 0 or positive_subgraph.numel() == 0 or negative_subgraph.numel() == 0:
            return torch.tensor(0.0, device=node_embedding.device)

        # 确保输入张量的维度正确
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
        if positive_subgraph.dim() == 1:
            positive_subgraph = positive_subgraph.unsqueeze(0)
        if negative_subgraph.dim() == 1:
            negative_subgraph = negative_subgraph.unsqueeze(0)

        # 计算正样本对的相似度
        pos_sim = self._compute_similarity(node_embedding, positive_subgraph) / self.temperature

        # 计算负样本对的相似度
        neg_sim = self._compute_similarity(node_embedding, negative_subgraph) / self.temperature

        # 计算InfoNCE损失
        logits = torch.cat([pos_sim, neg_sim], dim=0)
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float)
        labels[0] = 1.0  # 正样本对的标签为1

        # 计算softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=0, keepdim=True))

        # 计算InfoNCE损失
        loss = -(labels * log_prob).sum(dim=0) / labels.sum(dim=0)

        return loss

    def structural_contrast(self,
                            node_embedding: torch.Tensor,
                            positive_subgraph: torch.Tensor,
                            negative_subgraph: torch.Tensor) -> torch.Tensor:
        """计算结构InfoNCE损失"""
        # 检查输入是否为空
        if node_embedding.numel() == 0 or positive_subgraph.numel() == 0 or negative_subgraph.numel() == 0:
            return torch.tensor(0.0, device=node_embedding.device)

        # 确保输入张量的维度正确
        if node_embedding.dim() == 1:
            node_embedding = node_embedding.unsqueeze(0)
        if positive_subgraph.dim() == 1:
            positive_subgraph = positive_subgraph.unsqueeze(0)
        if negative_subgraph.dim() == 1:
            negative_subgraph = negative_subgraph.unsqueeze(0)

        # 计算正样本对的相似度
        pos_sim = self._compute_similarity(node_embedding, positive_subgraph) / self.temperature

        # 计算负样本对的相似度
        neg_sim = self._compute_similarity(node_embedding, negative_subgraph) / self.temperature

        # 计算InfoNCE损失
        logits = torch.cat([pos_sim, neg_sim], dim=0)
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float)
        labels[0] = 1.0  # 正样本对的标签为1

        # 计算softmax
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=0, keepdim=True))

        # 计算InfoNCE损失
        loss = -(labels * log_prob).sum(dim=0) / labels.sum(dim=0)

        return loss

    def forward(self,
                node_embeddings: torch.Tensor,
                temporal_pos_subgraphs: torch.Tensor,
                temporal_neg_subgraphs: torch.Tensor,
                structural_pos_subgraphs: torch.Tensor,
                structural_neg_subgraphs: torch.Tensor) -> torch.Tensor:
        """计算总InfoNCE损失"""
        # 检查输入是否为空
        if (node_embeddings.numel() == 0 or
                temporal_pos_subgraphs.numel() == 0 or
                temporal_neg_subgraphs.numel() == 0 or
                structural_pos_subgraphs.numel() == 0 or
                structural_neg_subgraphs.numel() == 0):
            return torch.tensor(0.0, device=node_embeddings.device)

        # 确保输入张量的维度正确
        if node_embeddings.dim() == 1:
            node_embeddings = node_embeddings.unsqueeze(0)
        if temporal_pos_subgraphs.dim() == 1:
            temporal_pos_subgraphs = temporal_pos_subgraphs.unsqueeze(0)
        if temporal_neg_subgraphs.dim() == 1:
            temporal_neg_subgraphs = temporal_neg_subgraphs.unsqueeze(0)
        if structural_pos_subgraphs.dim() == 1:
            structural_pos_subgraphs = structural_pos_subgraphs.unsqueeze(0)
        if structural_neg_subgraphs.dim() == 1:
            structural_neg_subgraphs = structural_neg_subgraphs.unsqueeze(0)

        # 计算时序InfoNCE损失
        temporal_loss = self.temporal_contrast(
            node_embeddings,
            temporal_pos_subgraphs,
            temporal_neg_subgraphs
        )

        # 计算结构InfoNCE损失
        structural_loss = self.structural_contrast(
            node_embeddings,
            structural_pos_subgraphs,
            structural_neg_subgraphs
        )

        # 计算总损失
        total_loss = (1 - self.beta) * temporal_loss + self.beta * structural_loss

        return total_loss