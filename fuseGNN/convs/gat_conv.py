import torch
from torch_geometric.nn import GATConv as geoGATConv
from fuseGNN.functional import fused_gar_agg, fused_gas_agg, csr2csc, coo2csr, gat_gar_edge_weight, gat_gas_edge_weight
from fuseGNN.functional.format import Coo2Csr
import gcnlib_gat
from torch.nn import Parameter
import torch.nn.functional as F
import torch_scatter
import math
from fuseGNN.functional import dropout as my_dropout

# behavior module with pure python

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
        
        
###############################################################################################
# Reference Module
###############################################################################################
class refGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, flow='target_to_source', return_mask=False, cached=False):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
            fused (bool, optional): If set to True, the gcnlib.gcn_aggregate_f_cuda will be used. Default: False,
            verify (bool, optional): If set to True, it will output the difference between fused and unfused version
        """
        super(refGATConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # The dense layer for Update stage. The total number of out_freatures is out_channels * heads
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels * heads, bias=bias)
        
        # The attention parameters
        self.att = Parameter(torch.Tensor(heads * out_channels, 2))
        
        glorot(self.dense.weight)
        glorot(self.att)
        # zeros(self.bias)
        
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        
        self.return_mask = return_mask
        
    
    def forward(self, x, edge_index, dp_mask=None, dp_mask_self=None):
        """
        Args:
            x (float32 [N(v), in_channels]) : matrix of feature vectors
            edge_index (int64 [2, N(v)]) : the list of edges
            edge_weight (float32 [N(v)], optional) : the weight on each edge, default: 1
        """
        # step 0: split the edge index and convert them to int32
        src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
        
        # step 1: Update, forward input features into linear layer
        x = self.dense(x)
        
        # Aggregation stage
        return self.propagate(x, src_index, tar_index)
        
    def propagate(self, feature, src_index, tar_index):
        # step 1: get edge weight
        e_pre = torch.matmul(feature, self.att)
        
        e_pre_src, e_pre_tar = torch.chunk(e_pre, chunks=2, dim=1)
        e_pre_src_expand = torch.index_select(e_pre_src, dim=0, index=src_index)
        e_pre_tar_expand = torch.index_select(e_pre_tar, dim=0, index=tar_index)
        e = e_pre_src_expand + e_pre_tar_expand
        e_self = e_pre_src + e_pre_tar
        e = torch.exp(F.leaky_relu(e, negative_slope=self.negative_slope))
        e_self = torch.exp(F.leaky_relu(e_self, negative_slope=self.negative_slope))
        e_sum = torch_scatter.scatter_add(src=e, index=tar_index, dim=0, dim_size=feature.size(0))
        e_sum += e_self
        alpha_self = e_self / e_sum
        e_sum = torch.index_select(e_sum, dim=0, index=tar_index)
        alpha = e / e_sum
        # apply dropout
        alpha, mask = my_dropout(alpha, self.dropout, self.training)
        alpha_self, mask_self = my_dropout(alpha_self, self.dropout, self.training)
        
        
        # step 1: scatter the feature vectors to the extended feature map of src
        src_fm = torch.index_select(feature, dim=0, index=src_index)
        
        # step 2: scatter the feature vectors to the extended feature map of tar
        tar_fm = torch.index_select(feature, dim=0, index=tar_index)
        
        extended_fm = src_fm * alpha
        
        # step 3: scatter to the output feature
        out = torch_scatter.scatter_add(src=extended_fm, index=tar_index, dim=0, dim_size=feature.size(0))
        
        # step 4: apply self-loops
        out += feature * alpha_self
        
        if not self.concat:
            out = out.view(-1, self.heads, self.out_channels)
            out = out.mean(dim=1).squeeze_()
        
        if self.return_mask:
            return out, mask, mask_self
        else:
            return out


class garGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, flow='target_to_source', return_mask=False,
                 cached=False):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
            fused (bool, optional): If set to True, the gcnlib.gcn_aggregate_f_cuda will be used. Default: False,
            verify (bool, optional): If set to True, it will output the difference between fused and unfused version
        """
        super(garGATConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # The dense layer for Update stage. The total number of out_freatures is out_channels * heads
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels * heads, bias=bias)
        
        # The attention parameters
        self.att = Parameter(torch.Tensor(heads * out_channels, 2))
        
        glorot(self.dense.weight)
        glorot(self.att)
        # zeros(self.bias)
        
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        
        self.cached = cached
        self.cached_tar_index_b = None
        self.cached_src_index_b = None
        self.cached_src_ptr = None
        
        self.return_mask = return_mask
        
    
    def forward(self, x, edge_index=None, dp_mask=None, dp_mask_self=None, tar_index_b=None,
                src_index_b=None, src_ptr=None):
        """
        Args:
            x (float32 [N(v), in_channels]) : matrix of feature vectors
            edge_index (int64 [2, N(v)]) : the list of edges
            edge_weight (float32 [N(v)], optional) : the weight on each edge, default: 1
        """
        # step 0: split the edge index and convert them to int32
        if (not self.cached) or (self.cached_tar_index_b is None):
            if src_ptr is None:  
                src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
                src_index = src_index.to(torch.int32)
                tar_index = tar_index.to(torch.int32)
                self.cached_tar_index_b, self.cached_src_index_b, self.cached_src_ptr, dp_mask = coo2csr(tar_index, src_index, 
                                                                                                         x.size(0), dp_mask, False)
            else:
                self.cached_tar_index_b = tar_index_b
                self.cached_src_index_b = src_index_b
                self.cached_src_ptr = src_ptr
        
        # step 1: Update, forward input features into linear layer
        x = self.dense(x)
        
        # Aggregation stage
        return self.propagate(x, dp_mask, dp_mask_self)
    
    def propagate(self, feature, dp_mask, dp_mask_self):
        if dp_mask is not None:
            dp_mask = dp_mask.view(-1, 1)
        e_pre = torch.matmul(feature, self.att)
        
        alpha_self, alpha = gat_gar_edge_weight(e_pre, self.cached_src_ptr, 
                                                self.cached_tar_index_b, self.cached_src_index_b, 
                                                self.negative_slope)
        
        # dropout on edge weight. 
        if dp_mask is not None and self.training: # If the mask is provided
            alpha = alpha * dp_mask
            alpha_self = alpha_self * dp_mask_self
        else: # Otherwise
            alpha = F.dropout(alpha, self.dropout, self.training)
            alpha_self = F.dropout(alpha_self, self.dropout, self.training)
        
        tar_ptr, src_index_f, alpha_f = csr2csc(self.cached_src_ptr.detach_(), self.cached_tar_index_b.detach_(), 
                                                alpha.detach(), feature.size(0))
        
        out = fused_gar_agg(feature=feature, src_index=src_index_f, tar_ptr=tar_ptr, 
                            edge_weight_f=alpha_f.detach_(), self_edge_weight=alpha_self,
                            tar_index=self.cached_tar_index_b, src_ptr=self.cached_src_ptr, edge_weight_b=alpha,
                            require_edge_weight=True)
        
        if not self.concat:
            out = out.view(-1, self.heads, self.out_channels)
            out = out.mean(dim=1).squeeze_()
        
        
        if self.return_mask:
            return out, dp_mask, dp_mask_self
        else:
            return out
        
        
class gasGATConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, flow='target_to_source', return_mask=False,
                 cached=False):
        """
        Args:
            in_channels (int): Size of each input sample
            out_channels (int): Size of each output sample.
            cached (bool, optional): if set to True, the layer will cache
                the computation of D^{-0.5}\hat{A}of D^{-0.5} on the first
                execution, and it will be used for further executions.
                This is only helpful in transductive learning
            bias (bool, optional): If set to True, there will be a bias
            flow (str): could be the following two conditions
                'source_to_target': edge_index[0] is the source nodes, [1] is the target nodes
                'target_to_source': edge_index[0] is the target nodes, [0] is the source nodes
            fused (bool, optional): If set to True, the gcnlib.gcn_aggregate_f_cuda will be used. Default: False,
            verify (bool, optional): If set to True, it will output the difference between fused and unfused version
        """
        super(gasGATConv, self).__init__()
        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']
        self.tid, self.sid = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        # The dense layer for Update stage. The total number of out_freatures is out_channels * heads
        self.dense = torch.nn.Linear(in_features=in_channels, out_features=out_channels * heads, bias=bias)
        
        # The attention parameters
        self.att = Parameter(torch.Tensor(heads * out_channels, 2))
        
        glorot(self.dense.weight)
        glorot(self.att)
        # zeros(self.bias)
        
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat
        self.cached = cached
        self.cached_src_index = None
        self.cached_tar_index = None
        
        self.return_mask = return_mask
        
    
    def forward(self, x, edge_index=None, dp_mask=None, dp_mask_self=None, src_index=None, tar_index=None):
        """
        Args:
            x (float32 [N(v), in_channels]) : matrix of feature vectors
            edge_index (int64 [2, N(v)]) : the list of edges
            edge_weight (float32 [N(v)], optional) : the weight on each edge, default: 1
        """
        # step 0: split the edge index and convert them to int32
        if (not self.cached) or (self.cached_src_index is None):
            if src_index is None:
                src_index, tar_index = (edge_index[self.sid], edge_index[self.tid])
                src_index = src_index.to(torch.int32)
                tar_index = tar_index.to(torch.int32)
            self.cached_src_index = src_index
            self.cached_tar_index = tar_index
        
        # step 1: Update, forward input features into linear layer
        x = self.dense(x)
        
        # Aggregation stage
        return self.propagate(x, dp_mask, dp_mask_self)
        
    def propagate(self, feature, dp_mask, dp_mask_self):
        
        e_pre = torch.matmul(feature, self.att)
        
        alpha_self, alpha = gat_gas_edge_weight(e_pre, self.cached_src_index, self.cached_tar_index, self.negative_slope)
        
        if dp_mask is not None and self.training: # If the mask is provided
            alpha = alpha * dp_mask
            alpha_self = alpha_self * dp_mask_self
        else: # Otherwise
            alpha = F.dropout(alpha, self.dropout, self.training)
            alpha_self = F.dropout(alpha_self, self.dropout, self.training)
        
        out = fused_gas_agg(feature=feature, src_index=self.cached_src_index, tar_index=self.cached_tar_index, 
                            edge_weight=alpha, self_edge_weight=alpha_self,
                            require_edge_weight=True)
        
        if not self.concat:
            out = out.view(-1, self.heads, self.out_channels)
            out = out.mean(dim=1).squeeze_()
        
        
        if self.return_mask:
            return out, dp_mask, dp_mask_self
        else:
            return out