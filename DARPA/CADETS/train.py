##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from kairos_utils import *
from config import *
from model import *
import config
import inspect

from graph_sampler import OptimizedGraphSampler, \
    DualBranchContrastiveLearning  # StructuralTemporalSampler, DualBranchContrastiveLearning

# Setting for logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(artifact_dir + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def seq_batches(data, batch_size):
    for i in range(0, data.num_events, batch_size):
        src = data.src[i:i + batch_size]
        dst = data.dst[i:i + batch_size]
        t = data.t[i:i + batch_size]
        msg = data.msg[i:i + batch_size]
        yield Batch(src=src, dst=dst, t=t, msg=msg)


class Batch:
    def __init__(self, src, dst, t, msg):
        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg
        self.num_events = len(src)


def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader,
          sampler,
          contrastive_learning
          ):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    total_link_loss = 0
    total_contrastive_loss = 0

    for batch in tqdm(seq_batches(train_data, batch_size=BATCH), desc="Training"):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # 修改过之后
        # 在每个batch开始时重置assoc数组
        assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # 使用优化后的采样器进行批量采样
        temporal_pos_embeddings, temporal_neg_embeddings, \
            structural_pos_embeddings, structural_neg_embeddings = sampler.batch_sample(src, t)

        # Compute contrastive loss using triplet margin loss
        contrastive_loss = contrastive_learning.forward(
            z[assoc[src]],  # Center node embeddings
            temporal_pos_embeddings,
            temporal_neg_embeddings,
            structural_pos_embeddings,
            structural_neg_embeddings
        )

        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        link_loss = criterion(y_pred, y_true)

        # Combined loss with equal weighting
        # loss = link_loss + contrastive_loss
        loss = link_loss + 0.1*contrastive_loss.detach() #+ weight_decay * (torch.norm(gnn.parameters()) + torch.norm(link_pred.parameters()))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        # 后来的加的
        loss.backward()
        optimizer.step()
        memory.detach()

        total_loss += float(loss) * batch.num_events
        print(f'Loss: {loss:.4f}')
        print(f'Link Loss: {link_loss:.4f}')
        print(f'Contrastive Loss: {contrastive_loss:.4f}')

    # 在训练循环的适当位置
    # sampler.save_all_caches()

    return total_loss / train_data.num_events


# 原文链接：https: // blog.csdn.net / weixin_46552666 / article / details / 140076098
"""
def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader,
          sampler,
          contrastive_learning
          ):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    total_link_loss = 0
    total_contrastive_loss = 0

    for batch in tqdm(seq_batches(train_data, batch_size=BATCH), desc="Training"):
        print("第三圈")
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
        #print("n_id")
        #print(n_id)
        n_id, edge_index, e_id = neighbor_loader(n_id)

        # 修改过之后
        # 在每个batch开始时重置assoc数组
        # assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)

        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Perform contrastive learning
        temporal_pos_embeddings = []
        temporal_neg_embeddings = []
        structural_pos_embeddings = []
        structural_neg_embeddings = []
        count = 0
        for node in src:
            # Temporal sampling (η-BFS)
            # print(f'count in src: {count}')
            pos_emb, _ = sampler.eta_bfs_sampling(node, t[0], edge_index, train_data.t[e_id], train_data.msg[e_id])
            neg_emb, _ = sampler.eta_bfs_sampling(node, t[0]-1000, edge_index, train_data.t[e_id], train_data.msg[e_id])
            temporal_pos_embeddings.append(pos_emb.mean(dim=0))
            temporal_neg_embeddings.append(neg_emb.mean(dim=0))

            # Structural sampling (ε-DFS)
            pos_emb, _ = sampler.epsilon_dfs_sampling(node, t[0], edge_index, train_data.t[e_id], train_data.msg[e_id])
            neg_emb, _ = sampler.epsilon_dfs_sampling((node + 1) % max_node_num, t[0], edge_index, train_data.t[e_id], train_data.msg[e_id])
            structural_pos_embeddings.append(pos_emb.mean(dim=0))
            structural_neg_embeddings.append(neg_emb.mean(dim=0))
            count = count + 1

        # Stack embeddings
        temporal_pos_embeddings = torch.stack(temporal_pos_embeddings)
        temporal_neg_embeddings = torch.stack(temporal_neg_embeddings)
        structural_pos_embeddings = torch.stack(structural_pos_embeddings)
        structural_neg_embeddings = torch.stack(structural_neg_embeddings)

        # Compute contrastive loss using triplet margin loss
        contrastive_loss = contrastive_learning.forward(
            z[assoc[src]],  # Center node embeddings
            temporal_pos_embeddings,
            temporal_neg_embeddings,
            structural_pos_embeddings,
            structural_neg_embeddings
        )

        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[node_embedding_dim:-node_embedding_dim], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        link_loss = criterion(y_pred, y_true)

        # Combined loss with equal weighting
        loss = link_loss + contrastive_loss

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)


        total_loss += float(loss) * batch.num_events
        print(f'Loss: {loss:.4f}')
        print(f'Link Loss: {link_loss:.4f}')
        print(f'Contrastive Loss: {contrastive_loss:.4f}')
    return total_loss / train_data.num_events
"""


def load_train_data():
    graph_4_2 = torch.load(graphs_dir + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(graphs_dir + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(graphs_dir + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]


def init_models(node_feat_size):
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        node_state_dim,
        time_dim,
        message_module=IdentityMessage(node_feat_size, node_state_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=node_state_dim,
        out_channels=edge_dim,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    out_channels = len(include_edge_type)
    link_pred = LinkPredictor(in_channels=edge_dim, out_channels=out_channels).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=lr, eps=eps, weight_decay=weight_decay)

    neighbor_loader = LastNeighborLoader(max_node_num, size=neighbor_size, device=device)

    return memory, gnn, link_pred, optimizer, neighbor_loader


if __name__ == "__main__":
    logger.info("Start logging.")

    # Load data for training
    train_data = load_train_data()

    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # Initialize the sampler and contrastive learning modules
    sampler = OptimizedGraphSampler(
        memory=memory,
        neighbor_loader=neighbor_loader,
        eta=2,
        epsilon=2,
        k=2,
        device=device,
        cache_dir='/home/lbj/kairos-main/kairos-main/DARPA/CADETS_E3/sampling_cache',
        num_workers=8
    )

    contrastive_learning = DualBranchContrastiveLearning(
        temperature=0.1,  # 0.1,
        beta=0.5
    )
    # 预计算采样结果
    # 在训练循环开始前
    all_nodes = []
    all_timestamps = []
    for g in train_data:
        all_nodes.extend(g.src.tolist())
        all_timestamps.extend(g.t.tolist())

    # 预计算采样结果
    sampler.precompute_sampling(all_nodes, all_timestamps)

    # train the model
    for epoch in tqdm(range(1, epoch_num + 1)):
        print("第一圈")
        for g in train_data:
            print("第二圈")
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader,
                sampler=sampler,
                contrastive_learning=contrastive_learning
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Save the trained model
    model = [memory, gnn, link_pred, neighbor_loader]

    os.system(f"mkdir -p {models_dir}")
    # torch.save(model, f"{models_dir}/models.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model, f"{models_dir}/models_{timestamp}.pt")

    config_items = {k: v for k, v in config.__dict__.items() if
                    not k.startswith('__') and not inspect.ismodule(v) and not inspect.isfunction(v)}
    config_save_path = f"{artifact_dir}config_{timestamp}.txt"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        for k, v in config_items.items():
            f.write(f"{k} = {v}\n")
    logger.info(f"Config saved to {config_save_path}")
