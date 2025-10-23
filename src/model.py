import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    实现位置编码模块。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x 的形状: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    实现多头自注意力机制。
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v 的形状: [batch_size, n_heads, seq_len, d_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def forward(self, x, mask=None):
        # x 的形状: [batch_size, seq_len, d_model]
        batch_size = x.size(0)

        # 1. 线性变换并切分头
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算缩放点积注意力
        # attention_output 形状: [batch_size, n_heads, seq_len, d_k]
        attention_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        # 3. 合并头
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        
        # 4. 最终线性变换
        output = self.out(attention_output)
        return output

class PositionwiseFeedForward(nn.Module):
    """
    实现逐位置前馈网络。
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 的形状: [batch_size, seq_len, d_model]
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    """
    实现一个 Transformer Encoder Block。
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x 的形状: [batch_size, seq_len, d_model]
        
        # 1. 多头注意力 + Add & Norm
        attn_output = self.attention(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 2. 前馈网络 + Add & Norm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

class TransformerEncoder(nn.Module):
    """
    实现完整的 Transformer Encoder 模型。
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src, src_mask=None):
        # src 的形状: [batch_size, seq_len]
        
        # 1. 嵌入和位置编码
        # embed_src 形状: [batch_size, seq_len, d_model]
        embed_src = self.embedding(src) * math.sqrt(self.d_model)
        
        # PyTorch 的 Transformer 模块期望输入形状为 [seq_len, batch_size, d_model]
        # 我们将保持 [batch_size, seq_len, d_model] 并在需要时调整
        
        # 调整为 [seq_len, batch_size, d_model] 以适应 PositionalEncoding
        pos_encoded_src = self.pos_encoder(embed_src.transpose(0, 1)).transpose(0, 1)
        
        # 2. 通过 Encoder Blocks
        encoder_output = pos_encoded_src
        for layer in self.layers:
            encoder_output = layer(encoder_output, src_mask)
            
        # 3. 输出层
        output = self.output_layer(encoder_output)
        return output

if __name__ == "__main__":
    # --- 模型组件测试 ---
    print("--- 开始模型组件测试 ---")
    
    # 超参数
    VOCAB_SIZE = 10000
    D_MODEL = 128
    N_HEADS = 4
    D_FF = 512
    N_LAYERS = 2
    DROPOUT = 0.1
    BATCH_SIZE = 32
    SEQ_LEN = 35

    # 创建一个假的输入张量
    test_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    # 1. 测试 PositionalEncoding
    print("\n1. 测试 PositionalEncoding...")
    embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
    pos_encoder = PositionalEncoding(D_MODEL, DROPOUT)
    embedded_input = embedding(test_input).transpose(0, 1) # -> [seq_len, batch_size, d_model]
    pos_encoded_output = pos_encoder(embedded_input)
    print(f"  PositionalEncoding 输入形状: {embedded_input.shape}")
    print(f"  PositionalEncoding 输出形状: {pos_encoded_output.shape}")
    assert pos_encoded_output.shape == (SEQ_LEN, BATCH_SIZE, D_MODEL)
    print("  PositionalEncoding 测试通过。")

    # 2. 测试 MultiHeadAttention
    print("\n2. 测试 MultiHeadAttention...")
    multi_head_attn = MultiHeadAttention(D_MODEL, N_HEADS, DROPOUT)
    # 输入形状: [batch_size, seq_len, d_model]
    attn_input = pos_encoded_output.transpose(0, 1) 
    attn_output = multi_head_attn(attn_input)
    print(f"  MultiHeadAttention 输入形状: {attn_input.shape}")
    print(f"  MultiHeadAttention 输出形状: {attn_output.shape}")
    assert attn_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("  MultiHeadAttention 测试通过。")

    # 3. 测试 PositionwiseFeedForward
    print("\n3. 测试 PositionwiseFeedForward...")
    ffn = PositionwiseFeedForward(D_MODEL, D_FF, DROPOUT)
    ffn_output = ffn(attn_output)
    print(f"  PositionwiseFeedForward 输入形状: {attn_output.shape}")
    print(f"  PositionwiseFeedForward 输出形状: {ffn_output.shape}")
    assert ffn_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("  PositionwiseFeedForward 测试通过。")

    # 4. 测试 EncoderBlock
    print("\n4. 测试 EncoderBlock...")
    encoder_block = EncoderBlock(D_MODEL, N_HEADS, D_FF, DROPOUT)
    encoder_block_output = encoder_block(attn_input)
    print(f"  EncoderBlock 输入形状: {attn_input.shape}")
    print(f"  EncoderBlock 输出形状: {encoder_block_output.shape}")
    assert encoder_block_output.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)
    print("  EncoderBlock 测试通过。")

    # 5. 测试完整的 TransformerEncoder
    print("\n5. 测试完整的 TransformerEncoder...")
    model = TransformerEncoder(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, DROPOUT)
    model_output = model(test_input)
    print(f"  TransformerEncoder 输入形状: {test_input.shape}")
    print(f"  TransformerEncoder 输出形状: {model_output.shape}")
    assert model_output.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    print("  TransformerEncoder 测试通过。")

    print("\n--- 所有模型组件测试成功 ---")
