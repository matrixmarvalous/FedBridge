

from torch import nn

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, varlen_embedding_lookup, \
    get_varlen_pooling_list

DEFAULT_GROUP_NAME = "default_group"

def _idx_range(feature_index, name):
    idx = feature_index[name]
    if isinstance(idx, tuple):

        if len(idx) == 2:
            start, second = idx

            if isinstance(second, int) and second >= 1 and second < 1e6:

                length = second
                end = start + length
            else:
                end = second
        elif len(idx) >= 3:

            start, end = idx[0], idx[1]
        else:
            start = idx[0]
            end = start + 1
    else:
        start = int(idx)
        end = start + 1
    return int(start), int(end)


def _peek_unique(x):
    try:
        import torch
        vals = torch.unique(x).cpu().view(-1).tolist()
        vals = vals[:50]
        return vals
    except Exception:
        return None


import torch

def _safe_embedding_call(emb, x_idx, feat_name):
    """
    emb: nn.Embedding (或类似)
    x_idx: LongTensor of indices (batch, 1) 或 (batch,)
    feat_name: 字符串，便于打印定位
    """

    num = emb.num_embeddings if hasattr(emb, "num_embeddings") else emb.weight.shape[0]

    try:
        x_min = int(torch.min(x_idx).item())
        x_max = int(torch.max(x_idx).item())
    except Exception:
        x_min, x_max = None, None


    bad_mask = (x_idx < 0) | (x_idx >= num)
    if torch.any(bad_mask):
        bad_vals = x_idx[bad_mask]

        examples = bad_vals.flatten()[:20].tolist()
        raise RuntimeError(
            f"[Embedding OOR] feat={feat_name} | "
            f"num_embeddings={num} | allowed=[0, {num-1}] | "
            f"min_seen={x_min} | max_seen={x_max} | "
            f"bad_examples={examples}"
        )
    return emb(x_idx)


class SparseFeatP(SparseFeat):
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        return super(SparseFeatP, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                               embedding_name, group_name)

    def __init__(self, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, padding_idx=None):
        self.padding_idx = padding_idx


def get_dataset_columns(dim_user, dim_action, num_user, num_action, envname="VirtualTB-v0"):
    user_columns, action_columns, feedback_columns = [], [], []
    has_user_embedding, has_action_embedding, has_feedback_embedding = None, None, None
    if envname == "VirtualTB-v0":
        user_columns = [DenseFeat("feat_user", 88)]
        action_columns = [DenseFeat("feat_item", 27)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = True
        has_action_embedding = True
        has_feedback_embedding = True
    else: # for kuairecenv, coat, yahoo
        user_columns = [SparseFeatP("feat_user", num_user, embedding_dim=dim_user)]
        action_columns = [SparseFeatP("feat_item", num_action, embedding_dim=dim_action)]
        feedback_columns = [DenseFeat("feat_feedback", 1)]
        has_user_embedding = False
        has_action_embedding = False
        has_feedback_embedding = True

    return user_columns, action_columns, feedback_columns, \
           has_user_embedding, has_action_embedding, has_feedback_embedding


def input_from_feature_columns(X, feature_columns, embedding_dict, feature_index, support_dense: bool, device):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    if not support_dense and len(dense_feature_columns) > 0:
        raise ValueError(
            "DenseFeat is not supported in dnn_feature_columns")
        
    
    for feat in sparse_feature_columns:
        s, e = feature_index[feat.name]

        if feat.embedding_name.lower() in ("gender","age","age_range","occupation","user_id","item_id") or feat.name.startswith("feat"):
            assert (e - s) == 1, f"[inputs] {feat.name} width must be 1, got {e-s}"
    

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]
    
    
    sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                  varlen_sparse_feature_columns)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                           varlen_sparse_feature_columns, device)

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeatP), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse,
                                           padding_idx=feat.padding_idx)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )


    for tensor in embedding_dict.values():
        if tensor.padding_idx is None:
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        else:
            nn.init.normal_(tensor.weight[:tensor.padding_idx], mean=0, std=init_std)
            nn.init.normal_(tensor.weight[tensor.padding_idx+1:], mean=0, std=init_std)

    return embedding_dict.to(device)


