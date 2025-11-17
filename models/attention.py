"""
Attention-based Temporal Aggregation Module
Implements various attention mechanisms for aggregating temporal features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, query_len, embed_dim)
            key: (batch, key_len, embed_dim)
            value: (batch, value_len, embed_dim)
            mask: Optional attention mask
        Returns:
            output: (batch, query_len, embed_dim)
            attention_weights: (batch, num_heads, query_len, key_len)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for temporal aggregation
    Learns to weight different time steps based on their importance
    """
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Optional mask for valid timesteps (batch, time)
        Returns:
            pooled: Aggregated tensor (batch, input_dim)
            attention_weights: Attention scores (batch, time)
        """
        # Compute attention scores
        attn_scores = self.attention(x).squeeze(-1)  # (batch, time)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, time)
        
        # Weighted sum
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (batch, input_dim)
        
        return pooled, attn_weights


class SelfAttentionPooling(nn.Module):
    """
    Self-attention based pooling with multi-head attention
    """
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionPooling, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        
        # Learnable query for pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Optional mask for valid timesteps
        Returns:
            pooled: Aggregated tensor (batch, input_dim)
            attention_weights: Attention scores
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, input_dim)
        
        # Apply self-attention with query
        pooled, attn_weights = self.self_attn(query, x, x, mask)
        
        # Squeeze time dimension
        pooled = pooled.squeeze(1)  # (batch, input_dim)
        
        # Layer normalization
        pooled = self.norm(pooled)
        
        return pooled, attn_weights


class TemporalAttentionAggregator(nn.Module):
    """
    Advanced temporal aggregation with multiple attention mechanisms
    Combines max pooling, average pooling, and attention pooling
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, use_multi_pool=True):
        super(TemporalAttentionAggregator, self).__init__()
        
        self.input_dim = input_dim
        self.use_multi_pool = use_multi_pool
        
        # Attention pooling
        self.attn_pool = AttentionPooling(input_dim, hidden_dim)
        
        # Self-attention pooling
        self.self_attn_pool = SelfAttentionPooling(input_dim, num_heads)
        
        if use_multi_pool:
            # Fusion layer for combining different pooling strategies
            # 4 * input_dim: [attention, self-attention, max, avg]
            self.fusion = nn.Sequential(
                nn.Linear(input_dim * 4, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            # Simple fusion of two attention mechanisms
            self.fusion = nn.Sequential(
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU()
            )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch, time, input_dim)
            mask: Optional mask for valid timesteps
        Returns:
            aggregated: Aggregated tensor (batch, input_dim)
            attention_weights: Dictionary of attention weights
        """
        # Attention pooling
        attn_pooled, attn_weights = self.attn_pool(x, mask)
        
        # Self-attention pooling
        self_attn_pooled, self_attn_weights = self.self_attn_pool(x, mask)
        
        if self.use_multi_pool:
            # Max pooling
            max_pooled = torch.max(x, dim=1)[0]  # (batch, input_dim)
            
            # Average pooling
            if mask is not None:
                # Masked average
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                masked_x = x * mask_expanded
                avg_pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                avg_pooled = torch.mean(x, dim=1)  # (batch, input_dim)
            
            # Concatenate all pooling results
            pooled = torch.cat([attn_pooled, self_attn_pooled, max_pooled, avg_pooled], dim=-1)
        else:
            # Only use attention mechanisms
            pooled = torch.cat([attn_pooled, self_attn_pooled], dim=-1)
        
        # Fuse different pooling strategies
        aggregated = self.fusion(pooled)
        
        # Return aggregated features and attention weights
        weights = {
            'attention': attn_weights,
            'self_attention': self_attn_weights
        }
        
        return aggregated, weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test attention modules
    batch_size = 4
    seq_len = 32
    input_dim = 256
    
    print("Testing AttentionPooling...")
    x = torch.randn(batch_size, seq_len, input_dim)
    attn_pool = AttentionPooling(input_dim)
    pooled, weights = attn_pool(x)
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\nTesting SelfAttentionPooling...")
    self_attn_pool = SelfAttentionPooling(input_dim, num_heads=8)
    pooled, weights = self_attn_pool(x)
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\nTesting TemporalAttentionAggregator...")
    aggregator = TemporalAttentionAggregator(input_dim, use_multi_pool=True)
    aggregated, weights_dict = aggregator(x)
    print(f"Input shape: {x.shape}")
    print(f"Aggregated shape: {aggregated.shape}")
    print(f"Attention weights keys: {weights_dict.keys()}")
    
    print("\nTesting with mask...")
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len//2:] = 0  # Mask second half
    aggregated, weights_dict = aggregator(x, mask)
    print(f"Masked aggregated shape: {aggregated.shape}")
