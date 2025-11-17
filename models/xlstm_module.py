"""
xLSTM (Extended LSTM) Module
Implements mLSTM (matrix LSTM) and sLSTM (scalar LSTM) variants
Based on the xLSTM architecture for improved long-term memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class sLSTMCell(nn.Module):
    """
    Scalar LSTM (sLSTM) Cell with exponential gating
    Enhanced version of standard LSTM with stabilized gating
    """
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Forget gate
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output gate
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Cell state
        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Stabilization parameters
        self.m_i = nn.Parameter(torch.ones(1, hidden_size))
        self.m_f = nn.Parameter(torch.ones(1, hidden_size))
        
    def forward(self, x, states):
        """
        Args:
            x: Input tensor (batch, input_size)
            states: Tuple of (h, c, n, m) where
                h: hidden state (batch, hidden_size)
                c: cell state (batch, hidden_size)
                n: normalizer state (batch, hidden_size)
                m: max state (batch, hidden_size)
        Returns:
            h_new: New hidden state
            (h_new, c_new, n_new, m_new): New states
        """
        h, c, n, m = states
        
        # Compute gates
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))  # Input gate
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))  # Forget gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))  # Output gate
        
        # Compute cell candidate
        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h))
        
        # Exponential gating with stabilization
        i_exp = torch.exp(i * self.m_i)
        f_exp = torch.exp(f * self.m_f)
        
        # Update cell state with stabilization
        c_new = f_exp * c + i_exp * c_tilde
        n_new = f_exp * n + i_exp
        m_new = torch.max(f + m, i)
        
        # Stabilized cell state
        c_stabilized = c_new / (n_new + 1e-6)
        
        # Output
        h_new = o * torch.tanh(c_stabilized)
        
        return h_new, (h_new, c_new, n_new, m_new)


class mLSTMCell(nn.Module):
    """
    Matrix LSTM (mLSTM) Cell
    Uses matrix-valued memory for enhanced capacity
    """
    def __init__(self, input_size, hidden_size, head_dim=32):
        super(mLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = hidden_size // head_dim
        
        assert hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        # Input gate
        self.W_i = nn.Linear(input_size, hidden_size)
        
        # Forget gate
        self.W_f = nn.Linear(input_size, hidden_size)
        
        # Output gate
        self.W_o = nn.Linear(input_size, hidden_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, states):
        """
        Args:
            x: Input tensor (batch, input_size)
            states: Tuple of (C, n, h) where
                C: Matrix memory (batch, hidden_size, head_dim)
                n: Normalizer (batch, hidden_size)
                h: Hidden state (batch, hidden_size)
        Returns:
            h_new: New hidden state
            (C_new, n_new, h_new): New states
        """
        C, n, h = states
        batch_size = x.size(0)
        
        # Compute query, key, value
        q = self.W_q(x)  # (batch, hidden_size)
        k = self.W_k(x)  # (batch, hidden_size)
        v = self.W_v(x)  # (batch, hidden_size)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim)  # (batch, num_heads, head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)
        
        # Compute gates
        i = torch.sigmoid(self.W_i(x))  # Input gate
        f = torch.sigmoid(self.W_f(x))  # Forget gate
        o = torch.sigmoid(self.W_o(x))  # Output gate
        
        # Reshape gates for broadcasting
        i = i.view(batch_size, self.num_heads, self.head_dim)
        f = f.view(batch_size, self.num_heads, self.head_dim)
        o = o.view(batch_size, self.num_heads, self.head_dim)
        
        # Reshape C for multi-head
        C = C.view(batch_size, self.num_heads, self.head_dim, self.head_dim)
        n = n.view(batch_size, self.num_heads, self.head_dim)
        
        # Update matrix memory
        # C_new = f * C + i * (v @ k^T)
        kv = torch.einsum('bhd,bhe->bhde', v, k)  # (batch, num_heads, head_dim, head_dim)
        C_new = f.unsqueeze(-1) * C + i.unsqueeze(-1) * kv
        
        # Update normalizer
        n_new = f * n + i * k
        
        # Retrieve from memory
        # h = C @ q / n
        h_retrieved = torch.einsum('bhde,bhe->bhd', C_new, q)  # (batch, num_heads, head_dim)
        h_retrieved = h_retrieved / (n_new.unsqueeze(-1) + 1e-6)
        
        # Apply output gate
        h_new = o * torch.tanh(h_retrieved)
        
        # Reshape back
        h_new = h_new.reshape(batch_size, self.hidden_size)
        C_new = C_new.reshape(batch_size, self.hidden_size, self.head_dim)
        n_new = n_new.reshape(batch_size, self.hidden_size)
        
        # Layer normalization
        h_new = self.norm(h_new)
        
        return h_new, (C_new, n_new, h_new)


class sLSTM(nn.Module):
    """
    Multi-layer sLSTM module
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(sLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Create layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.cells.append(sLSTMCell(input_dim, hidden_size))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, states=None):
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)
            states: Initial states (optional)
        Returns:
            output: Output tensor (batch, seq_len, hidden_size)
            final_states: Final states for each layer
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states if not provided
        if states is None:
            states = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                c = torch.zeros(batch_size, self.hidden_size, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
                m = torch.zeros(batch_size, self.hidden_size, device=x.device)
                states.append((h, c, n, m))
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Pass through all layers
            for layer_idx in range(self.num_layers):
                h_t, states[layer_idx] = self.cells[layer_idx](x_t, states[layer_idx])
                x_t = h_t
                
                # Apply dropout between layers
                if layer_idx < self.num_layers - 1 and self.dropout_layer is not None:
                    x_t = self.dropout_layer(x_t)
            
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        
        return output, states


class mLSTM(nn.Module):
    """
    Multi-layer mLSTM module
    """
    def __init__(self, input_size, hidden_size, num_layers=1, head_dim=32, dropout=0.2):
        super(mLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.dropout = dropout
        
        # Create layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            self.cells.append(mLSTMCell(input_dim, hidden_size, head_dim))
        
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, states=None):
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)
            states: Initial states (optional)
        Returns:
            output: Output tensor (batch, seq_len, hidden_size)
            final_states: Final states for each layer
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize states if not provided
        if states is None:
            states = []
            for _ in range(self.num_layers):
                C = torch.zeros(batch_size, self.hidden_size, self.head_dim, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                states.append((C, n, h))
        
        # Process sequence
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Pass through all layers
            for layer_idx in range(self.num_layers):
                h_t, states[layer_idx] = self.cells[layer_idx](x_t, states[layer_idx])
                x_t = h_t
                
                # Apply dropout between layers
                if layer_idx < self.num_layers - 1 and self.dropout_layer is not None:
                    x_t = self.dropout_layer(x_t)
            
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        
        return output, states


if __name__ == "__main__":
    # Test xLSTM modules
    batch_size = 4
    seq_len = 32
    input_size = 256
    hidden_size = 256
    
    print("Testing sLSTM...")
    x = torch.randn(batch_size, seq_len, input_size)
    slstm = sLSTM(input_size, hidden_size, num_layers=2)
    output, states = slstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size})")
    
    print("\nTesting mLSTM...")
    mlstm = mLSTM(input_size, hidden_size, num_layers=2, head_dim=32)
    output, states = mlstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size})")
