# %% [markdown]
# # MultiSAGE on Tripartite Graph

# %%
from collections import Counter
from pathlib import Path
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import dgl.nn.pytorch as gnn
import dgl.function as fn
from dgl.nn.functional import edge_softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset

# %% [markdown]
# ## Graph Construction

# %%
datapath = Path('..', 'data')

# %%
with open(datapath / 'dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# %%
dataset.keys()

# %%
ing_rec = []
rec_ing = []
rec_cui = []
cui_rec = []

for recipe, (ingredients, cuisine) in enumerate(zip(dataset['X_train'], dataset['y_train'])):
    ing_rec.extend((ingredient, recipe) for ingredient in ingredients)
    rec_ing.extend((recipe, ingredient) for ingredient in ingredients)
    rec_cui.append((recipe, cuisine))
    cui_rec.append((cuisine, recipe))

graph = dgl.heterograph({
    ('ingredient', 'i2r', 'recipe'): ing_rec,
    ('recipe', 'r2i', 'ingredient'): rec_ing,
    ('recipe', 'r2c', 'cuisine'): rec_cui,
    ('cuisine', 'c2r', 'recipe'): cui_rec,
})

# %%
graph.ndata['h'] = {
    'ingredient': torch.randn((graph.num_nodes('ingredient'), 512)),
    'recipe': torch.randn((graph.num_nodes('recipe'), 512)),
    'cuisine': torch.randn((graph.num_nodes('cuisine'), 512)),
}

# %%
graph.edata

# %% [markdown]
# ## Define Model

# %% [markdown]
# ### Sampler

# %%


def compact_and_copy(frontier, seeds):
    block = dgl.to_block(g=frontier, dst_nodes=seeds)
    for col, data in frontier.edata.items():
        if col == dgl.EID:
            continue
        block.edata[col] = data[block.edata[dgl.EID]]
    return block

# %%


def assign_simple_node_features(ndata, g, ntype, assign_id=False):
    for col in g.nodes[ntype].data.keys():
        if not assign_id and col == dgl.NID:
            continue
        induced_nodes = ndata[dgl.NID]
        ndata[col] = g.nodes[ntype].data[col][induced_nodes]

# %%


def assign_features_to_blocks(blocks, g, ntype):
    assign_simple_node_features(blocks[0].srcdata, g, ntype)
    assign_simple_node_features(blocks[-1].dstdata, g, ntype)

# %%


def sample_context_blocks(g, blocks, context_dicts, ntype, ctype):
    context_blocks = []
    for block, context_dict in zip(blocks, context_dicts):
        context_ids = []
        inner_to_global_id = {}
        dom_context_dict, pair_context_dict = context_dict
        for inner, glob in zip(block.nodes(ntype).tolist(), block.ndata[dgl.NID][ntype].tolist()):
            inner_to_global_id[inner] = glob
        for src, dst in zip(block.edges()[0].tolist(), block.edges()[1].tolist()):
            if (inner_to_global_id[src], inner_to_global_id[dst]) in pair_context_dict:
                context_ids.append(pair_context_dict[(inner_to_global_id[src], inner_to_global_id[dst])])
            elif (inner_to_global_id[dst], inner_to_global_id[src]) in pair_context_dict:
                context_ids.append(pair_context_dict[(inner_to_global_id[dst], inner_to_global_id[src])])
            elif inner_to_global_id[dst] in dom_context_dict:
                context_ids.append(dom_context_dict[inner_to_global_id[dst]])
            else:
                eids = block.edge_ids(src, dst)
                block.remove_edges(eids)
        sample_graph = g.subgraph({ctype: context_ids})
        context_blocks.append(sample_graph.nodes[ctype].data)
    return context_blocks

# %%


class ItemToItemBatchSampler(IterableDataset):
    def __init__(self, g, user_type, item_type, batch_size):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.user_to_item_etype = list(g.metagraph()[user_type][item_type])[0]
        self.item_to_user_etype = list(g.metagraph()[item_type][user_type])[0]
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            heads = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            result = dgl.sampling.random_walk(
                self.g,
                heads,
                metapath=[self.item_to_user_etype, self.user_to_item_etype])
            tails = result[0][:, 2]
            neg_tails = torch.randint(0, self.g.number_of_nodes(self.item_type), (self.batch_size,))
            mask = (tails != -1)
            yield heads[mask], tails[mask], neg_tails[mask]

# %%


class RandomWalkNeighborSampler(object):
    def __init__(self, G, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, metapath=None, weight_column='weights'):
        assert G.device == dgl.backend.cpu(), "Graph must be on CPU."
        self.G = G
        self.weight_column = weight_column
        self.num_random_walks = num_random_walks
        self.num_neighbors = num_neighbors
        self.num_traversals = num_traversals

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError('Metapath must be specified if the graph is homogeneous.')
            metapath = [G.canonical_etypes[0]]
        start_ntype = G.to_canonical_etype(metapath[0])[0]
        end_ntype = G.to_canonical_etype(metapath[-1])[-1]
        if start_ntype != end_ntype:
            raise ValueError('The metapath must start and end at the same node type.')
        self.ntype = start_ntype

        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath * num_traversals
        restart_prob = np.zeros(self.metapath_hops * num_traversals)
        restart_prob[self.metapath_hops::self.metapath_hops] = termination_prob
        self.restart_prob = dgl.backend.zerocopy_from_numpy(restart_prob)

    def _make_context_dict(self, paths):
        dom_context_dict = {}
        pair_context_dict = {}

        # make pair context dict
        for path in paths.tolist():
            if path[1] != -1:
                if (path[0] != -1) and (path[2] != -1):
                    context = path[1]
                    pair = (path[0], path[2])
                    pair_context_dict[pair] = context
            if path[3] != -1:
                if (path[2] != -1) and (path[4] != -1):
                    context = path[3]
                    pair = (path[2], path[4])
                    pair_context_dict[pair] = context

        # make context for single nodes
        for item_nodes, ctx_nodes in zip(paths[:, [0, 2, 4]].tolist(), paths[:, [1, 3]].tolist()):
            for item in item_nodes:
                if item == -1:
                    continue
                for ctx in ctx_nodes:
                    if ctx == -1:
                        continue
                    else:
                        if item in dom_context_dict:
                            if ctx in dom_context_dict[item]:
                                dom_context_dict[item][ctx] += 1
                            else:
                                dom_context_dict[item][ctx] = 1
                        else:
                            dom_context_dict[item] = {}
                            dom_context_dict[item][ctx] = 1

        # set dorminant context for dst nodes
        for k, v in dom_context_dict.items():
            dom_context_dict[k] = Counter(v).most_common(1)[0][0]

        return (dom_context_dict, pair_context_dict)

    # pylint: disable=no-member
    def __call__(self, seed_nodes):
        seed_nodes = dgl.utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        seed_nodes = dgl.backend.repeat(seed_nodes, self.num_random_walks, 0)
        paths, hi = dgl.sampling.random_walk(
            self.G, seed_nodes, metapath=self.full_metapath, restart_prob=self.restart_prob)
        src = dgl.backend.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        dst = dgl.backend.repeat(paths[:, 0], self.num_traversals, 0)
        src_mask = (src != -1)
        src = dgl.backend.boolean_mask(src, src_mask)
        dst = dgl.backend.boolean_mask(dst, src_mask)
        context_dicts = self._make_context_dict(paths)

        # count the number of visits and pick the K-most frequent neighbors for each node
        neighbor_graph = dgl.convert.heterograph(
            {(self.ntype, '_E', self.ntype): (src, dst)},  # data dict
            {self.ntype: self.G.number_of_nodes(self.ntype)}  # num node dict
        )
        neighbor_graph = dgl.transform.to_simple(neighbor_graph, return_counts=self.weight_column)
        counts = neighbor_graph.edata[self.weight_column]
        neighbor_graph = dgl.sampling.select_topk(neighbor_graph, self.num_neighbors, self.weight_column)
        selected_counts = dgl.backend.gather_row(counts, neighbor_graph.edata[dgl.EID])
        neighbor_graph.edata[self.weight_column] = selected_counts
        return neighbor_graph, context_dicts


class PinSAGESampler(RandomWalkNeighborSampler):
    def __init__(self, G, ntype, other_type, num_traversals, termination_prob,
                 num_random_walks, num_neighbors, weight_column='weights'):
        metagraph = G.metagraph()
        fw_etype = list(metagraph[ntype][other_type])[0]
        bw_etype = list(metagraph[other_type][ntype])[0]
        super().__init__(G, num_traversals,
                         termination_prob, num_random_walks, num_neighbors,
                         metapath=[fw_etype, bw_etype], weight_column=weight_column)


# %%
class NeighborSampler(object):
    def __init__(self, g, user_type, item_type, random_walk_length, random_walk_restart_prob,
                 num_random_walks, num_neighbors, num_layers):
        self.g = g
        self.user_type = user_type
        self.item_type = item_type
        self.samplers = [
            PinSAGESampler(g, item_type, user_type, random_walk_length,
                           random_walk_restart_prob, num_random_walks, num_neighbors)
            for _ in range(num_layers)]

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        context_dicts = []
        for sampler in self.samplers:
            frontier, context_dict = sampler(seeds)
            if heads is not None:
                # edge ids node pointing to itself
                eids = frontier.edge_ids(torch.cat([heads, heads]), torch.cat([tails, neg_tails]), return_uv=True)[2]
                if len(eids) > 0:
                    frontier = dgl.remove_edges(frontier, eids)  # remove edge if the node pointing to itself
            block = compact_and_copy(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
            context_dicts.insert(0, context_dict)
        return blocks, context_dicts

    def sample_from_item_pairs(self, heads, tails, neg_tails):
        # Create a graph with positive connections only and another graph with negative
        pos_graph = dgl.graph(
            (heads, tails),
            num_nodes=self.g.number_of_nodes(self.item_type))
        neg_graph = dgl.graph(
            (heads, neg_tails),
            num_nodes=self.g.number_of_nodes(self.item_type))

        # remove isolated nodes and re-indexing all nodes and edges
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        seeds = pos_graph.ndata[dgl.NID]  # all node ids mapping to global graph g

        # extract 2-hop neighbor MFG structure dataset for message passing
        blocks, context_dicts = self.sample_blocks(seeds, heads, tails, neg_tails)
        return pos_graph, neg_graph, blocks, context_dicts

# %%


class PinSAGECollator(object):
    def __init__(self, sampler, g, ntype, ctype):
        self.sampler = sampler
        self.ntype = ntype
        self.ctype = ctype
        self.g = g

    def collate_train(self, batches):
        # batched graph infos from item2item random walk batcher
        heads, tails, neg_tails = batches[0]

        # construct multilayer neighborhood via PinSAGE
        pos_graph, neg_graph, blocks, context_dicts = self.sampler.sample_from_item_pairs(heads, tails, neg_tails)
        context_blocks = sample_context_blocks(self.g, blocks, context_dicts, self.ntype, self.ctype)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return pos_graph, neg_graph, blocks, context_blocks

    def collate_test(self, samples):
        batch = torch.LongTensor(samples)
        blocks, context_dicts = self.sampler.sample_blocks(batch)
        context_blocks = sample_context_blocks(self.g, blocks, context_dicts, self.ntype, self.ctype)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return blocks, context_blocks

    def collate_point(self, index_id):
        point = torch.LongTensor([index_id])
        blocks, context_dicts = self.sampler.sample_blocks(point)
        context_blocks = sample_context_blocks(self.g, blocks, context_dicts, self.ntype, self.ctype)
        assign_features_to_blocks(blocks, self.g, self.ntype)

        return blocks, context_blocks

# %% [markdown]
# ### Layers & Nets

# %%


class GATLayer(nn.Module):
    def __init__(self, input_dims):
        super(GATLayer, self).__init__()
        self.additive_attn_fc = nn.Linear(3 * input_dims, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.additive_attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        x = torch.cat([edges.src['z_src'], edges.dst['z_t'], edges.data['z_c']], dim=1)
        attention = self.additive_attn_fc(x)
        return {'attn': F.leaky_relu(attention)}

    def forward(self, block):
        block.apply_edges(self.edge_attention)
        attention = edge_softmax(block, block.edata['attn'])
        return attention


# %%
class MultiHeadGATLayer(nn.Module):
    def __init__(self, input_dims, num_heads, merge='mean'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(input_dims))
        self.merge = merge

    def forward(self, block):
        head_outs = [attn_head(block) for attn_head in self.heads]
        if self.merge == 'mean':
            return torch.mean(torch.stack(head_outs), 0)
        else:  # concatenate
            return torch.cat(head_outs, dim=0)

# %%


class MultiSAGEConv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, gat_num_heads, act=F.relu):
        super().__init__()
        self.multi_head_gat_layer = MultiHeadGATLayer(input_dims, gat_num_heads)
        self.act = act
        self.Q = nn.Linear(input_dims, hidden_dims)
        self.W = nn.Linear(input_dims + hidden_dims, output_dims)
        self.dropout = nn.Dropout(0.5)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Q.weight, gain=gain)
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def _transfer_raw_input(self, edges):
        return {'z_src_c': torch.mul(edges.src['z_src'], edges.data['z_c']),
                'z_t_c': torch.mul(edges.dst['z_t'], edges.data['z_c'])}

    def _node_integration(self, edges):
        return {'neighbors': edges.data['z_src_c'] * edges.data['a_mean'],
                'targets': edges.data['z_t_c'] * edges.data['a_mean']}

    def forward(self, block, h, context_node, attn_index=None):
        h_src, h_dst = h
        with block.local_scope():
            # transfer raw input feature
            z_src = self.act(self.Q(self.dropout(h_src)))
            z_c = self.act(context_node)
            block.srcdata['z_src'] = z_src
            block.dstdata['z_t'] = h_dst
            block.edata['z_c'] = z_c

            # getting attention
            attention = self.multi_head_gat_layer(block)
            if attn_index is not None:  # attn_index : index of attention which not in context id
                attention[attn_index] = 0
            block.edata['a_mean'] = attention

            # aggregation
            block.apply_edges(self._transfer_raw_input)
            block.apply_edges(self._node_integration)
            block.update_all(fn.copy_e('neighbors', 'm'), fn.sum('m', 'ns'))
            block.update_all(fn.copy_e('targets', 'm'), fn.sum('m', 'ts'))

            # normalize for context query
            if attn_index is not None:
                neighbor = block.dstdata['ns'] / (attention.shape[0] - sum(attn_index).item())
                target = block.dstdata['ts'] / (attention.shape[0] - sum(attn_index).item())
            else:
                neighbor = block.dstdata['ns'] / attention.shape[0]
                target = block.dstdata['ts'] / attention.shape[0]

            # normalize
            z = self.act(self.W(self.dropout(torch.cat([neighbor, target], 1))))
            z_norm = z.norm(2, 1, keepdim=True)
            z_norm = torch.where(z_norm == 0, torch.tensor(1.).to(z_norm), z_norm)
            z = z / z_norm
            return z

# %%


class MultiSAGENet(nn.Module):
    def __init__(self, hidden_dims, n_layers, gat_num_heads):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(MultiSAGEConv(hidden_dims, hidden_dims, hidden_dims, gat_num_heads))

    def forward(self, blocks, h, context_blocks, attn_index=None):
        for idx, (layer, block, context_node) in enumerate(zip(self.convs, blocks, context_blocks)):
            if (attn_index is not None) and (idx == 1):
                h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
                h = layer(block, (h, h_dst), context_node, attn_index)
            else:
                h_dst = h[:block.number_of_nodes('DST/' + block.ntypes[0])]
                h = layer(block, (h, h_dst), context_node)
        return h

# %%


class ItemToItemScorer(nn.Module):
    def __init__(self, full_graph, ntype):
        super().__init__()
        n_nodes = full_graph.number_of_nodes(ntype)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def _add_bias(self, edges):
        bias_src = self.bias[edges.src[dgl.NID]]
        bias_dst = self.bias[edges.dst[dgl.NID]]
        return {'s': edges.data['s'] + bias_src + bias_dst}

    def forward(self, item_item_graph, h):
        with item_item_graph.local_scope():
            item_item_graph.ndata['h'] = h
            item_item_graph.apply_edges(fn.u_dot_v('h', 'h', 's'))
            item_item_graph.apply_edges(self._add_bias)
            pair_score = item_item_graph.edata['s']
        return pair_score


# %% [markdown]
# ### Model

# %%
class MultiSAGEModel(nn.Module):
    def __init__(
        self,
        full_graph: dgl.DGLHeteroGraph,
        ntype: str,
        c1type: str,
        c2type: str,
        hidden_dims: int,
        n_layers: int,
        gat_num_heads: int,
    ):
        super().__init__()
        self.multisage = MultiSAGENet(hidden_dims, n_layers, gat_num_heads)
        self.node_scorer = ItemToItemScorer(full_graph, ntype)
        self.context1_scorer = ItemToItemScorer(full_graph, c1type)

    def forward(
        self,
        pos_graph,
        neg_graph,
        blocks,
        context_blocks,
        level,
    ):
        if level == 1:
            scorer = self.node_scorer
        elif level == 2:
            scorer = self.context1_scorer
        else:
            raise ValueError
        h_item = self.get_representation(blocks, context_blocks)
        pos_score = scorer(pos_graph, h_item)
        neg_score = scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_representation(
        self,
        blocks,
        context_blocks,
        context_id=None
    ):
        if context_id:
            return self.get_context_query(blocks, context_blocks, context_id)
        else:
            # h_item = self.nodeproj(blocks[0].srcdata)
            # h_item_dst = self.nodeproj(blocks[-1].dstdata)
            # z_c = self.contextproj(context_blocks[0])
            # z_c_dst = self.contextproj(context_blocks[-1])
            h_item = blocks[0].srcdata['h']
            h_item_dst = blocks[-1].dstdata['h']
            z_c = context_blocks[0]['h']
            z_c_dst = context_blocks[-1]['h']

            h = h_item_dst + self.multisage(blocks, h_item, (z_c, z_c_dst))
            return h

    def get_context_query(
        self,
        blocks,
        context_blocks,
        context_id
    ):
        # check sub-graph contains context id
        context_id = context_blocks[-1]['_ID'][0].item()
        print(context_id)
        print(context_blocks[-1]['_ID'])
        context_index = (context_id == context_blocks[-1]['_ID']).nonzero(as_tuple=True)[0]
        if context_index.size()[0] == 0:  # if context id not in sub-graph, only random sample context using for repr
            print("context not in sub graph")
            return self.get_representation(blocks, context_blocks)
        else:  # if context id in sub-graph, get MultiSAGE's context query
            print("execute context query")
            attn_index = torch.ones(context_blocks[-1]['_ID'].shape[0], dtype=bool)
            attn_index[context_index] = False
            h_item = self.nodeproj(blocks[0].srcdata)
            h_item_dst = self.nodeproj(blocks[-1].dstdata)
            z_c = self.contextproj(context_blocks[0])
            z_c_dst = self.contextproj(context_blocks[-1])
            h = h_item_dst + self.multisage(blocks, h_item, (z_c, z_c_dst), attn_index)
            return h

# %% [markdown]
# ## Training


# %%
random_walk_length = 2
random_walk_restart_prob = 0.5
num_random_walks = 10
num_neighbors = 5
num_layers = 2
gat_num_heads = 3
hidden_dims = 512
batch_size = 256
num_epochs = 5
num_workers = 0
lr = 3e-5
k = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
print(device.type)

# %%


def train():
    primary_batch_sampler = ItemToItemBatchSampler(
        graph, 'recipe', 'ingredient', batch_size)
    primary_neighbor_sampler = NeighborSampler(
        graph,
        'recipe',
        'ingredient',
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers)
    primary_collator = PinSAGECollator(primary_neighbor_sampler, graph, 'ingredient', 'recipe')
    primary_dataloader = DataLoader(
        primary_batch_sampler,
        collate_fn=primary_collator.collate_train,
        num_workers=num_workers)
    primary_dataloader_test = DataLoader(
        torch.arange(graph.number_of_nodes('ingredient')),
        batch_size=batch_size,
        collate_fn=primary_collator.collate_test,
        num_workers=num_workers)

    secondary_batch_sampler = ItemToItemBatchSampler(
        graph, 'cuisine', 'recipe', batch_size)
    secondary_neighbor_sampler = NeighborSampler(
        graph,
        'cuisine',
        'recipe',
        random_walk_length,
        random_walk_restart_prob,
        num_random_walks,
        num_neighbors,
        num_layers)
    secondary_collator = PinSAGECollator(secondary_neighbor_sampler, graph, 'recipe', 'cuisine')
    secondary_dataloader = DataLoader(
        secondary_batch_sampler,
        collate_fn=secondary_collator.collate_train,
        num_workers=num_workers)
    secondary_dataloader_test = DataLoader(
        torch.arange(graph.number_of_nodes('recipe')),
        batch_size=batch_size,
        collate_fn=secondary_collator.collate_test,
        num_workers=num_workers)

    model = MultiSAGEModel(
        graph,
        'ingredient',
        'recipe',
        'cuisine',
        hidden_dims,
        num_layers,
        gat_num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch_id in range(num_epochs):
        for batch_id, (pos_graph, neg_graph, blocks, context_blocks) in enumerate(primary_dataloader):
            loss = model(pos_graph, neg_graph, blocks, context_blocks, level=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                print(f'num_epochs: {epoch_id} | batches_per_epoch {batch_id} | '
                      f'loss: {loss}')

        for batch_id, (pos_graph, neg_graph, blocks, context_blocks) in enumerate(secondary_dataloader):
            loss = model(pos_graph, neg_graph, blocks, context_blocks, level=2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 10 == 0:
                print(f'num_epochs: {epoch_id} | batches_per_epoch {batch_id} | '
                      f'loss: {loss}')

    model.eval()
    with torch.no_grad():
        h_item_batches = []
        for blocks, context_blocks in primary_dataloader_test:
            h_item_batch = model.get_representation(blocks, context_blocks)
            h_item_batches.append(h_item_batch)
        h_item = torch.cat(h_item_batches, 0)

        h_item_batches = []
        for blocks, context_blocks in secondary_dataloader_test:
            h_item_batch = model.get_representation(blocks, context_blocks)
            h_item_batches.append(h_item_batch)
        h_item = torch.cat(h_item_batches, 0)

    return model, h_item


# %%
model, h_item = train()

# %%
