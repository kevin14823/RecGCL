# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

from recbole_gnn.model.abstract_recommender import GeneralGraphRecommender
from recbole_gnn.model.layers import LightGCNConv


class RecGCL(GeneralGraphRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(RecGCL, self).__init__(config, dataset)
        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]
        
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type: the embedding size of the base model
        self.n_layers = config['n_layers']          # int type: the layer num of the base model
        self.reg_weight = config['reg_weight']      # float32 type: the weight decay for l2 normalization

        self.eps = config['eps']
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.hyper_layers = config['hyper_layers']

        self.alpha = config['alpha']

        self.proto_reg = config['proto_reg']
        self.k = config['num_clusters']

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_users, self.latent_dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_items, self.latent_dim)))
        self.G_u_list = [None] * (self.n_layers + 1)
        self.G_i_list = [None] * (self.n_layers + 1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0

        self.E_u_list = [None] * (self.n_layers + 1)
        self.E_i_list = [None] * (self.n_layers + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        
         # SVD-related attributes
        self.q = config['q']  # Number of singular values and vectors
        self.alpha_align = config['alpha_align']
        self.alpha_uniform = config['alpha_uniform']
        
        # get the normalized adjust matrix
        self.adj_norm = self.coo2tensor(self.create_adjust_matrix())

        # perform svd reconstruction
        svd_u, s, svd_v = torch.svd_lowrank(self.adj_norm, q=self.q)
        self.u_mul_s = svd_u @ (torch.diag(s))
        self.v_mul_s = svd_v @ (torch.diag(s))
        del s
        self.ut = svd_u.T
        self.vt = svd_v.T

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def e_step(self):
        user_embeddings = self.user_embedding.weight.detach().cpu().numpy()
        item_embeddings = self.item_embedding.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings)

    def run_kmeans(self, x):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        import faiss
        kmeans = faiss.Kmeans(d=self.latent_dim, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.
        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for layer_idx in range(max(self.n_layers, self.hyper_layers * 2)):
            all_embeddings = self.gcn_conv(all_embeddings, self.edge_index, self.edge_weight)
            embeddings_list.append(all_embeddings)
            self.E_u_list[layer_idx], self.E_i_list[layer_idx] = torch.split(all_embeddings, [self.n_users, self.n_items])

        lightgcn_all_embeddings = torch.stack(embeddings_list[:self.n_layers + 1], dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])
        return user_all_embeddings, item_all_embeddings, embeddings_list

    def ProtoNCE_loss(self, node_embedding, user, item):
        user_embeddings_all, item_embeddings_all = torch.split(
            node_embedding, [self.n_users, self.n_items]
        )
        
        # user part
        user_embeddings = user_embeddings_all[user]               
        norm_user_embeddings = F.normalize(user_embeddings, dim=1)
        user2cluster = self.user_2cluster[user]                   
        user2centroids = self.user_centroids[user2cluster]        
        pos_score_user = (norm_user_embeddings * user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.t())
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # item part
        item_embeddings = item_embeddings_all[item]               
        norm_item_embeddings = F.normalize(item_embeddings, dim=1)
        item2cluster = self.item_2cluster[item]                   
        item2centroids = self.item_centroids[item2cluster]        
        pos_score_item = (norm_item_embeddings * item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.t())
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)

        alignment_loss = ((norm_user_embeddings - norm_item_embeddings)**2).sum(dim=1).mean()

        batch_user_norm = F.normalize(user_embeddings, dim=1)
        dist_mat = torch.cdist(batch_user_norm, batch_user_norm, p=2)
        exp_neg_dist = torch.exp(-2 * (dist_mat ** 2))
        eye_mask = torch.eye(exp_neg_dist.size(0), device=exp_neg_dist.device)
        exp_neg_dist = exp_neg_dist * (1 - eye_mask)
        uniformity_loss_user = torch.log(exp_neg_dist.sum() / (batch_user_norm.size(0) ** 2))

        alignment_uniformity_loss = self.alpha_align * alignment_loss + self.alpha_uniform * uniformity_loss_user

        total_loss = proto_nce_loss + alignment_uniformity_loss
        return total_loss
    
    def calc_ssl_loss(self, E_u_norm, E_i_norm, user_list, pos_item_list):
        r"""Calculate the loss of self-supervised tasks.

        Args:
            E_u_norm (torch.Tensor): Ego embedding of all users in the original graph after forwarding.
            E_i_norm (torch.Tensor): Ego embedding of all items in the original graph after forwarding.
            user_list (torch.Tensor): List of the user.
            pos_item_list (torch.Tensor): List of positive examples.

        Returns:
            torch.Tensor: Loss of self-supervised tasks.
        """
        # calculate G_u_norm&G_i_norm
        for layer in range(1, self.n_layers + 1):
            # svd_adj propagation
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = self.u_mul_s @ vt_ei
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = self.v_mul_s @ ut_eu

        # aggregate across layer
        G_u_norm = sum(self.G_u_list)
        G_i_norm = sum(self.G_i_list)

        neg_score = torch.log(torch.exp(G_u_norm[user_list] @ E_u_norm.T / self.ssl_temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_norm[pos_item_list] @ E_i_norm.T / self.ssl_temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_norm[user_list] * E_u_norm[user_list]).sum(1) / self.ssl_temp, -5.0, 5.0)).mean() + (
            torch.clamp((G_i_norm[pos_item_list] * E_i_norm[pos_item_list]).sum(1) / self.ssl_temp, -5.0, 5.0)).mean()
        ssl_loss = -pos_score + neg_score
        return self.ssl_reg * ssl_loss

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        center_embedding = embeddings_list[0]
    
        ssl_loss = self.calc_ssl_loss(user_all_embeddings, item_all_embeddings, user, pos_item)
        proto_loss = self.ProtoNCE_loss(center_embedding, user, pos_item)

        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e, embedding_list = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)
    
    def create_adjust_matrix(self):
        r"""Get the normalized interaction matrix of users and items.

        Returns:
            coo_matrix of the normalized interaction matrix.
        """
        ratings = np.ones_like(self._user, dtype=np.float32)
        matrix = sp.csr_matrix(
            (ratings, (self._user, self._item)),
            shape=(self.n_users, self.n_items),
        ).tocoo()
        rowD = np.squeeze(np.array(matrix.sum(1)), axis=1)
        colD = np.squeeze(np.array(matrix.sum(0)), axis=0)
        for i in range(len(matrix.data)):
            matrix.data[i] = matrix.data[i] / pow(rowD[matrix.row[i]] * colD[matrix.col[i]], 0.5)
        return matrix

    def coo2tensor(self, matrix: sp.coo_matrix):
        r"""Convert coo_matrix to tensor.

        Args:
            matrix (scipy.coo_matrix): Sparse matrix to be converted.

        Returns:
            torch.sparse.FloatTensor: Transformed sparse matrix.
        """
        indices = torch.from_numpy(
            np.vstack((matrix.row, matrix.col)).astype(np.int64))
        values = torch.from_numpy(matrix.data)
        shape = torch.Size(matrix.shape)
        x = torch.sparse.FloatTensor(indices, values, shape).coalesce().to(self.device)
        return x
