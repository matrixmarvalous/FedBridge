# -*- coding: utf-8 -*-
# @Time    : 2021/10/6 9:58 下午
# @Author  : Chongming GAO
# @FileName: inputs.py

# from collections import namedtuple
from torch import nn

from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, varlen_embedding_lookup, \
    get_varlen_pooling_list

DEFAULT_GROUP_NAME = "default_group"

def _idx_range(feature_index, name):
    idx = feature_index[name]
    if isinstance(idx, tuple):
        # 常见两种： (start, length) 或 (start, end)
        if len(idx) == 2:
            start, second = idx
            # 如果 second 看起来像“长度”，就用 start:start+length；否则当成 end
            if isinstance(second, int) and second >= 1 and second < 1e6:
                # 这里无法 100% 断言是 length 还是 end；更稳妥的：若 second <= 64 这类典型 max_len，就当 length
                length = second
                end = start + length
            else:
                end = second
        elif len(idx) >= 3:
            # 极少见，这里取前两个
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
        vals = vals[:50]  # 只看前 50 个
        return vals
    except Exception:
        return None


# --- 顶部加 ---
import torch

def _safe_embedding_call(emb, x_idx, feat_name):
    """
    emb: nn.Embedding (或类似)
    x_idx: LongTensor of indices (batch, 1) 或 (batch,)
    feat_name: 字符串，便于打印定位
    """
    # 允许范围：[0, num_embeddings - 1]
    num = emb.num_embeddings if hasattr(emb, "num_embeddings") else emb.weight.shape[0]
    # 统计 min/max
    try:
        x_min = int(torch.min(x_idx).item())
        x_max = int(torch.max(x_idx).item())
    except Exception:
        x_min, x_max = None, None

    # 找到所有越界位置
    bad_mask = (x_idx < 0) | (x_idx >= num)
    if torch.any(bad_mask):
        bad_vals = x_idx[bad_mask]
        # 只打印前若干个例子避免刷屏
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
        # feedback_columns = [SparseFeat("feat_feedback", 11, embedding_dim=27)]
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
        
#     import torch
    
        
#     sparse_embedding_list = []
#     for feat in feature_columns:
#         # 仅对稀疏特征检查（你的报错行正是这个分支）
#         if isinstance(feat, SparseFeat):
#             idx = X[:, feature_index[feat.name]].long()
#             emb_layer = embedding_dict[feat.embedding_name]
#             try:
#                 sparse_embedding_list.append(emb_layer(idx))
#             except IndexError as e:
#                 with torch.no_grad():
#                     min_v = int(idx.min().item())
#                     max_v = int(idx.max().item())
#                     vocab = int(emb_layer.num_embeddings)
#                     bad_mask = (idx < 0) | (idx >= vocab)
#                     bad_pos = bad_mask.nonzero(as_tuple=True)[0].view(-1)
#                     sample_pos = bad_pos[:20].tolist()                 # 取前 20 个错误位置
#                     sample_vals = [int(idx[p].item()) for p in sample_pos]
#                     raise IndexError(
#                         f"[Embedding OOR] feature='{feat.name}', emb='{feat.embedding_name}', "
#                         f"allowed index range = [0, {vocab-1}], "
#                         f"observed min={min_v}, max={max_v}. "
#                         f"first_bad_positions={sample_pos}, first_bad_values={sample_vals}"
#                     ) from e



#     # 文件: src/core/util/inputs.py
#     import torch
#     # ... 其他 import 保持不变 ...

#     # 在 input_from_feature_columns(...) 内，替换之前 try/except 的块
#     sparse_embedding_list = []
#     for feat in feature_columns:
#         if isinstance(feat, SparseFeat):
#             idx = X[:, feature_index[feat.name]].long()
#             emb_layer = embedding_dict[feat.embedding_name]
#             try:
#                 sparse_embedding_list.append(emb_layer(idx))
#             except IndexError as e:
#                 with torch.no_grad():
#                     orig_shape = tuple(idx.shape)                # 可能是 [B] 或 [B, C]
#                     idx_flat = idx.view(-1)
#                     vocab = int(emb_layer.num_embeddings)
#                     min_v = int(idx_flat.min().item())
#                     max_v = int(idx_flat.max().item())
#                     bad_flat = (idx_flat < 0) | (idx_flat >= vocab)
#                     bad_ids  = bad_flat.nonzero(as_tuple=False).view(-1)

#                     # 采样前 20 个越界值
#                     sample_vals = idx_flat[bad_ids][:20].tolist()

#                     # 如果是二维，给出 (row, col) 位置；一维则给出扁平索引
#                     pos_info = ""
#                     if idx.dim() == 2:
#                         B, C = idx.shape
#                         rows = (bad_ids // C).tolist()
#                         cols = (bad_ids %  C).tolist()
#                         pos_info = f", first_bad_rc={list(zip(rows[:20], cols[:20]))}"
#                     else:
#                         pos_info = f", first_bad_pos={bad_ids[:20].tolist()}"

#                     raise IndexError(
#                         f"[Embedding OOR] feature='{feat.name}', emb='{feat.embedding_name}', "
#                         f"idx_shape={orig_shape}, allowed=[0,{vocab-1}], "
#                         f"observed min={min_v}, max={max_v}, "
#                         f"first_bad_values={sample_vals}{pos_info}"
#                     ) from e

    # 改为：
#     sparse_embedding_list = []
#     # for feat in sparse_feature_columns:
#     #     emb = embedding_dict[feat.embedding_name]
#     #     # 原位置替换
#     #     start, end = _idx_range(feature_index, feat.name)
#     #     x_slice = X[:, start:end].long()
#     #             # 在 for feat in sparse_feature_columns: 里，取完 x_slice 之后加：
#     print("==== FEATURE SLICES (sparse) ====")
#     for feat in sparse_feature_columns:
#         idx = feature_index[feat.name]
#         start, end = _idx_range(feature_index, feat.name)
#         print(f"{feat.name:>20}: idx={idx} -> cols=[{start},{end}) len={end-start} emb={feat.embedding_name}")
#     print("=================================")
#         if feat.embedding_name.lower() in ("gender", "age"):
#             print(f"[peek] feat={feat.embedding_name} cols=[{start},{end}) uniques={_peek_unique(x_slice)}")


    
    
#         if feat.embedding_name in ("gender", "age", "Age", "Gender"):
#             print(f"[peek] feat={feat.embedding_name} start={start} end={end} uniques={_peek_unique(x_slice)}")
#         # x_slice = X[:, feature_index[feat.name]: feature_index[feat.name] + 1].long()
#         # 关键：带检查的调用
#         out = _safe_embedding_call(emb, x_slice, feat.embedding_name)
#         sparse_embedding_list.append(out)

#     # --- build sparse embeddings (drop-in replacement) ---
#     sparse_embedding_list = []

#     # 可选：打印每个稀疏特征在 X 中的列区间（只打印一次，便于排查错位）
#     print("==== FEATURE SLICES (sparse) ====")
#     for _feat in sparse_feature_columns:
#         _idx = feature_index[_feat.name]
#         _s, _e = _idx_range(feature_index, _feat.name)
#         print(f"{_feat.name:>20}: idx={_idx} -> cols=[{_s},{_e}) len={_e-_s} emb={_feat.embedding_name}")
#     print("=================================")

#     for feat in sparse_feature_columns:
#         emb = embedding_dict[feat.embedding_name]
#         _log_feat_index_shape_once(feat.name, feature_index[feat.name])

#         start, end = _idx_range(feature_index, feat.name)
#         x_slice = X[:, start:end].long()

#         # 可选窥值：只在性别/年龄上看一下当前 batch 的唯一值
#         if feat.embedding_name.lower() in ("gender", "age"):
#             print(f"[peek] feat={feat.embedding_name} cols=[{start},{end}) uniques={_peek_unique(x_slice)}")

#         # 边界检查 + 真正的 embedding
#         out = _safe_embedding_call(emb, x_slice, feat.embedding_name)
#         sparse_embedding_list.append(out)
    # --- end replacement ---
        # 改为：
#         # 可选：打印每个稀疏特征在 X 中的列区间（只打印一次，便于排查错位）
#     print("==== FEATURE SLICES (sparse) ====")
#     for _feat in sparse_feature_columns:
#         _idx = feature_index[_feat.name]
#         _s, _e = _idx_range(feature_index, _feat.name)
#         print(f"{_feat.name:>20}: idx={_idx} -> cols=[{_s},{_e}) len={_e-_s} emb={_feat.embedding_name}")
#     print("=================================")
        
#     sparse_embedding_list = []
#     for feat in sparse_feature_columns:
#         emb = embedding_dict[feat.embedding_name]
#         # 原位置替换
#         start, end = _idx_range(feature_index, feat.name)
#         x_slice = X[:, start:end].long()
#                 # 在 for feat in sparse_feature_columns: 里，取完 x_slice 之后加：
#         if feat.embedding_name in ("gender", "age", "Age", "Gender"):
#             print(f"[peek] feat={feat.embedding_name} start={start} end={end} uniques={_peek_unique(x_slice)}")
#         # x_slice = X[:, feature_index[feat.name]: feature_index[feat.name] + 1].long()
#         # 关键：带检查的调用
#         out = _safe_embedding_call(emb, x_slice, feat.embedding_name)
#         sparse_embedding_list.append(out)
    for feat in sparse_feature_columns:
        s, e = feature_index[feat.name]
        # 单值稀疏特征必须 width=1（MovieLens 的 gender/age_range/occupation/user_id/item_id 都是）
        if feat.embedding_name.lower() in ("gender","age","age_range","occupation","user_id","item_id") or feat.name.startswith("feat"):
            assert (e - s) == 1, f"[inputs] {feat.name} width must be 1, got {e-s}"
    

    sparse_embedding_list = [embedding_dict[feat.embedding_name](
        X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
        feat in sparse_feature_columns]
    
    
    
    

#     sparse_embedding_list = []
#     for feat in sparse_feature_columns:
#         emb = embedding_dict[feat.embedding_name]  # nn.Embedding
#         idx = tensor_save[feat.name].long()        # 这是已经编码后的索引张量
    
#         # === Debug Guard: 打印越界来源 ===
#         max_idx = idx.max().item()
#         vocab = emb.num_embeddings
#         if max_idx >= vocab:
#             # 找出前几个越界样本，便于你快速定位
#             import torch
#             bad = torch.nonzero(idx >= vocab, as_tuple=False).view(-1)[:10]
#             print(f"[OOB] feature={feat.name} / emb={feat.embedding_name} | "
#                   f"max_idx={max_idx} >= vocab={vocab} | bad_examples={idx[bad].view(-1).tolist()}")
#             raise RuntimeError(f"Embedding OOB at feature {feat.name}: max_idx={max_idx} >= vocab={vocab}")

#     sparse_embedding_list.append(emb(idx))
#     ##

    sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                  varlen_sparse_feature_columns)
    varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                           varlen_sparse_feature_columns, device)

    dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                        dense_feature_columns]

    return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
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

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        if tensor.padding_idx is None:
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        else:
            nn.init.normal_(tensor.weight[:tensor.padding_idx], mean=0, std=init_std)
            nn.init.normal_(tensor.weight[tensor.padding_idx+1:], mean=0, std=init_std)

    return embedding_dict.to(device)


