import torch
import fgnn_agg
from fuseGNN.functional.format import csr2csc


# fused Aggregation phase with GAR model

class fusedGARAggV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, src_index, tar_ptr, edge_weight_f, 
                self_edge_weight, tar_index, src_ptr, edge_weight_b):
        out = fgnn_agg.fused_gar_f(feature, src_index, tar_ptr, edge_weight_f, self_edge_weight)
        ctx.save_for_backward(tar_index, src_ptr, edge_weight_b, self_edge_weight)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        tar_index, src_ptr, edge_weight_b, self_edge_weight = ctx.saved_tensors
        grad_features, _, _2 = fgnn_agg.fused_gar_b(grad_out, tar_index, tar_index, src_ptr, 
                                            edge_weight_b, self_edge_weight, False)
        return grad_features, None, None, None, None, None, None, None, None
    

class fusedGARAggV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, src_index, tar_ptr, edge_weight_f, 
                self_edge_weight, tar_index, src_ptr, edge_weight_b):
        out = fgnn_agg.fused_gar_f(feature, src_index, tar_ptr, edge_weight_f, self_edge_weight)
        ctx.save_for_backward(tar_index, src_ptr, edge_weight_b, self_edge_weight, feature)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        tar_index, src_ptr, edge_weight_b, self_edge_weight, feature = ctx.saved_tensors
        grad_features, grad_edge_weight, grad_weight_self = fgnn_agg.fused_gar_b(grad_out, feature, tar_index, src_ptr, 
                                                                                    edge_weight_b, self_edge_weight, 
                                                                                    True)
        return grad_features, None, None, None, grad_weight_self, None, None, grad_edge_weight, None


def fused_gar_agg(feature, src_index, tar_ptr, edge_weight_f, 
                self_edge_weight, tar_index, src_ptr, edge_weight_b,
                require_edge_weight=False):
    if require_edge_weight:
        return fusedGARAggV2.apply(feature, src_index, tar_ptr, edge_weight_f, 
                    self_edge_weight, tar_index, src_ptr, edge_weight_b)
    else:
        return fusedGARAggV1.apply(feature, src_index, tar_ptr, edge_weight_f, 
                self_edge_weight, tar_index, src_ptr, edge_weight_b)

class fusedGASAggV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, src_index, tar_index, edge_weight, self_edge_weight):
        ctx.save_for_backward(src_index, tar_index, edge_weight, self_edge_weight)
        out = fgnn_agg.fused_gas_f(feature, src_index, tar_index, edge_weight, self_edge_weight)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        src_index, tar_index, edge_weight, self_edge_weight = ctx.saved_tensors
        grad_features, _, _2 = fgnn_agg.fused_gas_b(grad_out, src_index, src_index, tar_index, 
                                                edge_weight, self_edge_weight, False)
        return grad_features, None, None, None, None


class fusedGASAggV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, src_index, tar_index, edge_weight, self_edge_weight):
        ctx.save_for_backward(src_index, tar_index, edge_weight, self_edge_weight, feature)
        out = fgnn_agg.fused_gas_f(feature, src_index, tar_index, edge_weight, self_edge_weight)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        src_index, tar_index, edge_weight, self_edge_weight, feature = ctx.saved_tensors
        grad_features, grad_edge_weight, grad_weight_self = fgnn_agg.fused_gas_b(grad_out, feature, src_index, 
                                                                                    tar_index, edge_weight, 
                                                                                    self_edge_weight, True)
        return grad_features, None, None, grad_edge_weight, grad_weight_self



def fused_gas_agg(feature, src_index, tar_index, edge_weight, self_edge_weight, require_edge_weight=False):
    if require_edge_weight:
        return fusedGASAggV2.apply(feature, src_index, tar_index, edge_weight, self_edge_weight)
    else:
        return fusedGASAggV1.apply(feature, src_index, tar_index, edge_weight, self_edge_weight)
