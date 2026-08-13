"""残差可预测性诊断 —— 判定 oracle 间隔是否可从路由信号中触达。

DTR 本质上是一个**受约束的残差预测器**：它试图用路由信号 u_d(x) 预测
静态混合 mix_d(x) 在该病例上的残差 (y_d - mix_d)，再据此调整专家权重。

所以真正的问题不是「我这个架构行不行」，而是「这些信号里有没有可预测的残差」。
本脚本用**完全不受约束**的回归器（岭 / 梯度提升 / 随机森林）在**折外**预测残差：
  · 若 OOF R² ≤ 0  →  任何建立在同一信号上的路由架构都不可能捕获 oracle 间隔。
    这是对**整类方法**的判定，不是对某一个实现的判定。
  · 若 OOF R² > 0  →  DTR 的负结果是实现问题，值得继续找架构。

诚实性声明（必须随结果一起报）：
  sft 列在 train 上是 sample-in（ck250 正是在这 98 个视频上微调的），
  因此含 sft 的变体对残差是**乐观**的。故同时报只用 ridge 专家（诚实 OOF）的变体。
  如果连乐观变体都拿不到正 R²，结论只会更强。
"""
import os, io, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ridge_moe as RM
import final_static_moe as FS
from surgmind_dtr import routing_signals, DIMS, K, PROJECT

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def oof_r2(U, r, seeds=5, model='ridge'):
    """折外 R²。多次重复分折取均值，避免单次划分运气。
    R² 用 1 - SSE/SST，SST 以**折内训练均值**为基准（不是全局均值），
    否则会把「预测全局均值」这种平凡行为算成正收益。"""
    scores = []
    for s in range(seeds):
        kf = KFold(5, shuffle=True, random_state=s)
        sse = sst = 0.0
        for tr, te in kf.split(U):
            sc = StandardScaler().fit(U[tr])
            Ztr, Zte = sc.transform(U[tr]), sc.transform(U[te])
            if model == 'ridge':
                m = RidgeCV(alphas=np.logspace(-2, 4, 25))
            elif model == 'gbr':
                m = GradientBoostingRegressor(random_state=s, n_estimators=200,
                                              max_depth=2, learning_rate=0.05)
            else:
                m = RandomForestRegressor(random_state=s, n_estimators=300,
                                          min_samples_leaf=3, n_jobs=-1)
            m.fit(Ztr, r[tr])
            p = m.predict(Zte)
            sse += float(((r[te] - p) ** 2).sum())
            sst += float(((r[te] - r[tr].mean()) ** 2).sum())      # 折内均值基准
        scores.append(1 - sse / sst)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo/feats_4src.npz'))
    ap.add_argument('--pca', type=int, default=64)
    ap.add_argument('--step', type=float, default=0.1)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--tag', default='resid_probe')
    ap.add_argument('--raw', action='store_true',
                    help='额外测一个变体：路由输入换成**原始特征的 PCA 成分**而非13维分歧信号。'
                         '用来回答「更丰富的输入能否预测残差」——这是分歧信号版留下的唯一开口。')
    ap.add_argument('--outdir', default=os.path.join(
        PROJECT, 'revision/evidence/results/sft_grpo'))
    a = ap.parse_args()

    # ---- 专家池：与 surgmind_dtr / final_static_moe 完全同一条构造路径
    src, Y, istr, names = RM.load_sources(a.feats, a.pca)
    snames = list(src)
    Ytr = Y[istr]
    ntr = [n for n, t in zip(names, istr) if t]
    fo = np.arange(len(Ytr)) % 5
    Otr = []
    for k in snames:
        o1, _, _ = RM.source_expert_preds(src[k][istr], Ytr, src[k][~istr], fo)
        Otr.append(o1)
    Xtr = np.hstack([src[k][istr] for k in snames])
    Xva = np.hstack([src[k][~istr] for k in snames])
    o1, _, _ = RM.source_expert_preds(Xtr, Ytr, Xva, fo)
    Otr.append(o1)
    snames = snames + ['joint']
    R = os.path.dirname(a.feats)
    rr = [json.loads(l) for l in io.open(
        os.path.join(R, 'pred_sftbalpd3_ck250_train_pd.jsonl'), encoding='utf-8')]
    pt = {r['video']: r['pred_levels'] for r in rr}
    Otr.append(np.array([pt[v] for v in ntr], dtype=float))
    snames = snames + ['sft']
    Ptr = np.stack(Otr, -1)
    print('专家池 %s | train %d' % (snames, len(Ytr)), flush=True)

    out = {'experts': snames, 'n_train': int(len(Ytr)), 'variants': {}}

    # 两个变体：含 sft（乐观，与 DTR 完全一致） / 不含 sft（诚实 OOF）
    for vname, keep in [('with_sft(乐观)', list(range(len(snames)))),
                        ('ridge_only(诚实OOF)', [i for i, s in enumerate(snames) if s != 'sft'])]:
        P = Ptr[:, :, keep]
        grids = list(FS.simplex(P.shape[-1], a.step))
        A0, _ = FS.pick_alpha(P, Ytr, np.arange(len(Ytr)), grids, 'qwk')
        U = routing_signals(P, A0, None)                 # (n,K,F)，与 DTR 同一函数
        mix = np.einsum('nkm,km->nk', P, A0)
        resid = Ytr - mix                                # 静态混合的残差
        print('\n===== %s | 信号维度 %d =====' % (vname, U.shape[-1]), flush=True)
        v = {'pi': {DIMS[i]: dict(zip([snames[j] for j in keep],
                                      [round(float(x), 2) for x in A0[i]])) for i in range(K)},
             'nfeat': int(U.shape[-1]), 'dims': {}}
        for i in range(K):
            r = resid[:, i]
            row = {'resid_std': round(float(r.std()), 4)}
            for mdl in ('ridge', 'gbr', 'rf'):
                mu, sd = oof_r2(U[:, i, :], r, a.seeds, mdl)
                row[mdl] = [round(mu, 4), round(sd, 4)]
            v['dims'][DIMS[i]] = row
            print('  %-12s resid_std=%.3f | ridge R²=%+.3f±%.3f  gbr R²=%+.3f±%.3f  rf R²=%+.3f±%.3f'
                  % (DIMS[i], row['resid_std'], row['ridge'][0], row['ridge'][1],
                     row['gbr'][0], row['gbr'][1], row['rf'][0], row['rf'][1]), flush=True)
        out['variants'][vname] = v

    # ---- 变体三：原始特征 PCA 成分作为路由输入（回答「更丰富输入是否可行」）
    if a.raw:
        P = Ptr
        grids = list(FS.simplex(P.shape[-1], a.step))
        A0, _ = FS.pick_alpha(P, Ytr, np.arange(len(Ytr)), grids, 'qwk')
        mix = np.einsum('nkm,km->nk', P, A0)
        resid = Ytr - mix
        Zraw = np.hstack([src[k][istr] for k in list(src)])   # 各源已各自 StandardScaler+PCA
        print('\n===== raw_feats(原始特征 PCA) | 输入维度 %d =====' % Zraw.shape[1], flush=True)
        v = {'nfeat': int(Zraw.shape[1]), 'dims': {}}
        for i in range(K):
            r = resid[:, i]
            row = {'resid_std': round(float(r.std()), 4)}
            for mdl in ('ridge', 'rf'):
                mu, sd = oof_r2(Zraw, r, a.seeds, mdl)
                row[mdl] = [round(mu, 4), round(sd, 4)]
            v['dims'][DIMS[i]] = row
            print('  %-22s resid_std=%.3f | ridge R²=%+.3f±%.3f  rf R²=%+.3f±%.3f'
                  % (DIMS[i], row['resid_std'], row['ridge'][0], row['ridge'][1],
                     row['rf'][0], row['rf'][1]), flush=True)
        out['variants']['raw_feats(原始特征PCA)'] = v

    p = os.path.join(a.outdir, 'resid_probe%s.json' % ('_raw' if a.raw else ''))
    json.dump(out, io.open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\n->', p, flush=True)


if __name__ == '__main__':
    main()
