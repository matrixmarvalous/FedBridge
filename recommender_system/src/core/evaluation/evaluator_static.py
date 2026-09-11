import numpy as np
import torch
from tqdm import tqdm

from src.core.evaluation.utils import get_feat_dominate_dict

def interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k, need_transform,
                           num_trajectory, item_feat_domination, remove_recommended, force_length=0, top_rate=0.8, draw_bar=False):
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0

    all_acts = []

    for i in tqdm(range(num_trajectory), desc=f"evaluate static method in {env.__str__()}"):
        [user_ori, item_init_ori], info = env.reset() # 
                # === 20250905 DEBUG: 检查 LabelEncoder 的 classes_ ===
        # if i == 0 and need_transform:
        #     try:
        #         u_classes = getattr(env.lbe_user, "classes_", None)
        #         it_classes = getattr(env.lbe_item, "classes_", None)
        #         print("[DEBUG] len(lbe_user.classes_):",
        #               None if u_classes is None else len(u_classes))
        #         if u_classes is not None and len(u_classes) > 0:
        #             # 打印前后各5个样本，防止刷屏
        #             print("[DEBUG] lbe_user.classes_ head:", u_classes[:5])
        #             print("[DEBUG] lbe_user.classes_ tail:", u_classes[-5:])
        #             in_user = bool(np.isin(user_ori, u_classes))
        #             print(f"[DEBUG] user_ori={user_ori} in lbe_user.classes_? {in_user}")
        #         else:
        #             print("[DEBUG] lbe_user.classes_ is None or empty!")

        #         print("[DEBUG] len(lbe_item.classes_):",
        #               None if it_classes is None else len(it_classes))
        #         if it_classes is not None and len(it_classes) > 0:
        #             print("[DEBUG] lbe_item.classes_ head:", it_classes[:5])
        #             print("[DEBUG] lbe_item.classes_ tail:", it_classes[-5:])
        #         else:
        #             print("[DEBUG] lbe_item.classes_ is None or empty!")
        #     except Exception as e:
        #         print("[DEBUG] error while probing classes_:", repr(e))

        
        if need_transform:
            user = env.lbe_user.inverse_transform([user_ori])[0]  #20250905报错检查
            # user = env.lbe_user.transform([user_ori])[0]
            # ##########
            # try:
            #     user = env.lbe_user.inverse_transform([user_ori])[0]
            # except Exception as e:
            #     print(f"[DEBUG] inverse_transform failed for user_ori={user_ori}")
            #     # 进一步检查：是否其实应该用 transform 而不是 inverse_transform
            #     try:
            #         _ = env.lbe_user.transform([user_ori])[0]
            #         print("[DEBUG] NOTE: transform() 可用，可能这里本应使用 transform 而不是 inverse_transform")
            #     except Exception as e2:
            #         print("[DEBUG] transform() 也失败：", repr(e2))
            #     raise
            #     ####################
        else:
            user = user_ori

        acts = []
        terminated = False
        while not terminated:
            recommended_id_transform, recommended_id_raw, reward_pred = model.recommend_k_item(
                user, dataset_val, k=k, is_softmax=is_softmax, epsilon=epsilon, is_ucb=is_ucb,
                recommended_ids=acts if remove_recommended else [])
            if need_transform:
                assert recommended_id_transform == env.lbe_item.transform([recommended_id_raw])[0]
            acts.append(recommended_id_transform)
            state, reward, terminated, truncated, info = env.step(recommended_id_transform)
            total_turns += 1
            # metric 1
            cumulative_reward += reward
            # metric 2
            click_loss = np.absolute(reward_pred - reward)
            total_click_loss += click_loss

            if terminated:
                if force_length > 0:  # do not end here
                    env.cur_user = user_ori
                    terminated = False
                else:
                    break
            if force_length > 0 and len(acts) >= force_length:
                terminated = True
                break

        all_acts.extend(acts)

    ctr = cumulative_reward / total_turns
    click_loss = total_click_loss / total_turns

    hit_item = len(set(all_acts))
    num_items = len(dataset_val.df_item_val)
    CV = hit_item / num_items
    CV_turn = hit_item / len(all_acts)

    # eval_result_RL = {"CTR": ctr, "click_loss": click_loss, "trajectory_len": total_turns / num_trajectory,
    #                   "trajectory_reward": cumulative_reward / num_trajectory}
    eval_result_RL = {
        "click_loss": click_loss,
        "CV": f"{CV:.5f}",
        "CV_turn": f"{CV_turn:.5f}",
        "ctr": ctr,
        "len_tra": total_turns / num_trajectory,
        "R_tra": cumulative_reward / num_trajectory}

    if need_transform:
        all_acts_origin = env.lbe_item.inverse_transform(all_acts)
    else:
        all_acts_origin = all_acts
    feat_dominate_dict = get_feat_dominate_dict(dataset_val.df_item_val, all_acts_origin, item_feat_domination, top_rate=top_rate)
    eval_result_RL.update(feat_dominate_dict)

    if remove_recommended:
        eval_result_RL = {f"NX_{force_length}_" + k: v for k, v in eval_result_RL.items()}

    return eval_result_RL


def test_static_model_in_RL_env(model, env, dataset_val, is_softmax=True, epsilon=0, is_ucb=False, k=1,
                                need_transform=False, num_trajectory=100, item_feat_domination=None, force_length=10, top_rate=0.8, draw_bar=False):
    eval_result_RL = {}

    eval_result_standard = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                                  need_transform, num_trajectory, item_feat_domination,
                                                  remove_recommended=False, force_length=0, top_rate=top_rate, draw_bar=draw_bar)

    # No overlap and end with the env rule
    eval_result_NX_0 = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                              need_transform, num_trajectory, item_feat_domination,
                                              remove_recommended=True, force_length=0, top_rate=top_rate, draw_bar=draw_bar)

    # No overlap and end with explicit length
    eval_result_NX_x = interactive_evaluation(model, env, dataset_val, is_softmax, epsilon, is_ucb, k,
                                              need_transform, num_trajectory, item_feat_domination,
                                              remove_recommended=True, force_length=force_length, top_rate=top_rate, draw_bar=draw_bar)

    eval_result_RL.update(eval_result_standard)
    eval_result_RL.update(eval_result_NX_0)
    eval_result_RL.update(eval_result_NX_x)

    return eval_result_RL


def test_taobao(model, env, epsilon=0):
    # test the model in the interactive system
    cumulative_reward = 0
    total_click_loss = 0
    total_turns = 0
    num_trajectory = 100

    for i in range(num_trajectory):
        features, info = env.reset()
        terminated = False
        while not terminated:
            res = model(torch.FloatTensor(features).to(model.device).unsqueeze(0)).to('cpu').squeeze()
            item_feat_predict = res[model.y_index['feat_item'][0]:model.y_index['feat_item'][1]]
            action = item_feat_predict.detach().numpy()

            if epsilon > 0 and np.random.random() < epsilon:
                # Activate epsilon greedy
                action = np.random.random(action.shape)

            reward_pred = res[model.y_index['y'][0]:model.y_index['y'][1]]

            features, reward, terminated, truncated, info = env.step(action)

            total_turns += 1

            # metric 1
            cumulative_reward += reward

            # metric 2
            click_loss = np.absolute(float(reward_pred.detach().numpy()) - reward)
            total_click_loss += click_loss

            if terminated:
                break

    ctr = cumulative_reward / total_turns  # /10
    click_loss = total_click_loss / total_turns

    # print('CTR: %.2f'.format(ctr))
    eval_result_RL = {"CTR": ctr,
                      "click_loss": click_loss,
                      "trajectory_len": total_turns / num_trajectory,
                      "trajectory_reward": cumulative_reward / num_trajectory}  # /10}

    return eval_result_RL

