import numpy as np
import torch
import torch.nn as nn
# (B,T,E,H)分别表示 批次/序列长度/输入词向量维度/隐藏维度
B, E, H = 1, 128, 3

def prepare_inputs():
    """
    使用 NumPy 准备输入数据
    构造一个词汇表比句子长度大的示例
    """
    np.random.seed(42)
    
    # 1. 定义词汇表 - V = 6（词汇表大小）
    # 这个词汇表包含了可能出现的所有词
    vocab = {
        "播放": 0, "周杰伦": 1, "的": 2, "稻香": 3, 
        "歌曲": 4, "音乐": 5  # 词汇表比我们要用的句子包含更多词
    }
    V = len(vocab)  # V = 6，词汇表大小
    
    # 2. 定义要处理的句子 - T = 4（序列长度）
    # 这个句子只用了词汇表中的一部分词
    tokens = ["播放", "周杰伦", "的", "稻香"]  # 只有4个词
    T = len(tokens)  # T = 4，序列长度
    
    
    # 将句子转换为ID序列
    ids = [vocab[t] for t in tokens]  # [0, 1, 2, 3]
    print(f"ID序列: {ids}")
    
    # 3. 创建词向量表 - 形状为 (V, E)
    # 为词汇表中的每个词创建向量，即使有些词在句子里没用到
    emb_table = np.random.randn(V, E).astype(np.float32)
    print(f"词向量表形状: {emb_table.shape} = (V={V}, E={E})")
    
    # 4. 从词向量表中提取句子的词向量
    # 注意：我们只提取句子中的词对应的向量
    sentence_embeddings = emb_table[ids]  # 形状变为 (T, E)
    print(f"句子词向量形状: {sentence_embeddings.shape} = (T={T}, E={E})")
    
    # 5. 添加batch维度 - 形状变为 (B, T, E)
    x_np = sentence_embeddings[None]  # 添加batch维度
    print(f"最终输入张量形状: {x_np.shape} = (B={B}, T={T}, E={E})")
    
    return tokens, x_np

def manual_rnn_np(x_np, U_np, W_np):
    """
    使用 NumPy 手动实现 RNN(无偏置): h_t = tanh(U x_t + W h_{t-1})
    
    Args:
        x_np: (B, T, E)
        U_np: (E, H)
        W_np: (H, H)
    Returns:
        outputs: (B, T, H)
        final_h、h_t: (B, H)
    """
    B_local, T_local, _ = x_np.shape
    # 初始化h_0，为零向量
    h_prev = np.zeros((B_local, H), dtype=np.float32)
    steps = []
    
    print(f"\nRNN处理过程:")
    print(f"输入形状: {x_np.shape}")
    print(f"序列长度 T = {T_local}")
    
    for t in range(T_local):
        x_t = x_np[:, t, :]  # 取出第t个时间步的输入
        print(f"时间步 {t}: 输入形状 {x_t.shape}")
        h_t = np.tanh(x_t @ U_np + h_prev @ W_np)
        steps.append(h_t)
        h_prev = h_t
    
    outputs = np.stack(steps, axis=1)
    print(f"输出形状: {outputs.shape} = (B={B_local}, T={T_local}, H={H})")
    
    return outputs, h_prev
def pytorch_run_forward(x,U,W):
    rnn=nn.Rnn(input_size=E,hidden_size=H,num_layers=1,
               nonlinearity='tanh',bias=False,batch_first=True,bidirectional=False)
    with torch.no_grad():
        # Pytorch内部存放的是转置后的权重
        rnn.weight_ih_l0.copy_(U.T)
        rnn.weight_ih_l0.copy_(W.T)
    y,h_n=rnn(x)
    return y,h_n.squeeze(0)
def main():
    _, x_np = prepare_inputs()

    # PyTorch 张量，用于 nn.RNN 模块
    x = torch.from_numpy(x_np).float()
    
    # 使用可学习参数 U, W（无偏置）
    torch.manual_seed(7)
    U = torch.randn(E, H)
    W = torch.randn(H, H)

    # --- 手写 RNN (使用 NumPy) ---
    U_np = U.detach().numpy()
    W_np = W.detach().numpy()

    print("--- 手写 RNN (NumPy) ---")
    out_manual_np, hT_manual_np = manual_rnn_numpy(x_np, U_np, W_np)
    print("输入形状:", x_np.shape)
    print("手写输出形状:", out_manual_np.shape)
    print("手写最终隐藏形状:", hT_manual_np.shape)

    print("\n--- PyTorch nn.RNN ---")
    out_torch, hT_torch = pytorch_rnn_forward(x, U, W)
    print("模块输出形状:", out_torch.shape)
    print("模块最终隐藏形状:", hT_torch.shape)

    print("\n--- 对齐验证 ---")
    # 将 NumPy 结果转回 PyTorch 张量以进行比较
    out_manual = torch.from_numpy(out_manual_np)
    hT_manual = torch.from_numpy(hT_manual_np)

    print("逐步输出一致:", torch.allclose(out_manual, out_torch, atol=1e-6))
    print("最终隐藏一致:", torch.allclose(hT_manual, hT_torch, atol=1e-6))
    print("最后一步输出等于最终隐藏:", torch.allclose(out_torch[:, -1, :], hT_torch, atol=1e-6))


if __name__ == "__main__":
    main()
    
    