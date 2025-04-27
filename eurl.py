import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

class EnhancedEURLTransformer(nn.Module):
    """
    Transformer model enhanced with EURL components:
    - Relational Latent Divergence (RLD)
    - Computational Entropy Loss (CEL)
    - Causal Feedback Reinforcement (CFR)
    - Meta-Learner for dynamic loss weighting
    - Multi-Scale Sparse Attention mechanism
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        sparse_rank: int = 64,  # Rank for sparse attention approximation
        kappa: float = 0.5,     # Hyperparameter for CEL
        c_const: float = 2.0,   # Hyperparameter for CEL
        eps: float = 1e-8,      # Small constant for numerical stability
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_encoder_layers
        self.kappa = kappa
        self.c_const = c_const
        self.eps = eps
        
        # Token embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Encoder layers with tracking of internal states
        self.layers = nn.ModuleList([
            EnhancedEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                sparse_rank=sparse_rank
            ) for _ in range(num_encoder_layers)
        ])
        
        # Final linear layer for prediction
        self.final_layer = nn.Linear(d_model, vocab_size)
        
        # Meta-learner for dynamic loss weighting
        self.meta_learner = MetaLearner(input_dim=3, hidden_dim=32)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_internal_states: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with collection of internal states for loss computation.
        
        Args:
            src: Input token indices [batch_size, seq_len]
            src_mask: Mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            return_internal_states: Whether to return internal states for loss computation
            
        Returns:
            output: Output logits
            internal_states: Dictionary of internal states for loss computation
        """
        internal_states = {
            'latent_embeddings': [],
            'attention_weights': [],
            'layer_losses': [],
            'layer_outputs': []
        } if return_internal_states else None
        
        # Embedding and positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Process through encoder layers
        for i, layer in enumerate(self.layers):
            layer_output = layer(
                x, src_mask=src_mask, 
                src_key_padding_mask=src_key_padding_mask,
                return_attention=return_internal_states
            )
            
            if return_internal_states:
                x, attn_weights, latent_emb = layer_output
                internal_states['latent_embeddings'].append(latent_emb)
                internal_states['attention_weights'].append(attn_weights)
                internal_states['layer_outputs'].append(x)
            else:
                x = layer_output
        
        # Final prediction
        output = self.final_layer(x)
        
        if return_internal_states:
            return output, internal_states
        else:
            return output
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        internal_states: Dict,
        reduction: str = 'mean'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Enhanced EURL loss.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target token indices [batch_size, seq_len]
            internal_states: Internal states from forward pass
            reduction: Reduction method ('mean', 'sum', or 'none')
            
        Returns:
            total_loss: The weighted sum of all loss components
            loss_components: Dictionary of individual loss components
        """
        # Base cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction
        )
        
        # Compute RLD loss
        rld_loss = self._compute_rld_loss(
            internal_states['latent_embeddings'],
            internal_states['attention_weights'],
            logits, targets
        )
        
        # Compute CEL loss
        cel_loss = self._compute_cel_loss(
            internal_states['attention_weights']
        )
        
        # Compute layer-wise losses for CFR
        for i, layer_output in enumerate(internal_states['layer_outputs']):
            layer_logits = self.final_layer(layer_output)
            layer_loss = F.cross_entropy(
                layer_logits.view(-1, layer_logits.size(-1)),
                targets.view(-1),
                reduction=reduction
            )
            internal_states['layer_losses'].append(layer_loss)
        
        # Compute CFR loss
        cfr_loss = self._compute_cfr_loss(internal_states['layer_losses'])
        
        # Use meta-learner to compute dynamic weights
        loss_values = torch.tensor([rld_loss.item(), cel_loss.item(), cfr_loss.item()])
        alpha, beta, gamma = self.meta_learner(loss_values)
        
        # Weighted sum of all losses
        total_loss = ce_loss + alpha * rld_loss + beta * cel_loss + gamma * cfr_loss
        
        # Return loss components for logging
        loss_components = {
            'ce_loss': ce_loss.item(),
            'rld_loss': rld_loss.item(),
            'cel_loss': cel_loss.item(),
            'cfr_loss': cfr_loss.item(),
            'alpha': alpha.item(),
            'beta': beta.item(),
            'gamma': gamma.item()
        }
        
        return total_loss, loss_components
    
    def _compute_rld_loss(
        self,
        latent_embeddings: list,
        attention_weights: list,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Relational Latent Divergence (RLD) loss.
        
        RLD = (1/N) ∑_{i,j} [GraphSim(z_i, A_ij) - TargetSim(y_i, ŷ_i)]
        """
        batch_size, seq_len = logits.shape[:2]
        
        # Calculate target similarity (using one-hot encoded targets)
        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pred_probs = F.softmax(logits, dim=-1)
        
        # Calculate cosine similarity between target and prediction
        target_sim = F.cosine_similarity(
            target_one_hot.view(-1, logits.size(-1)),
            pred_probs.view(-1, logits.size(-1)),
            dim=1
        ).view(batch_size, seq_len)
        
        rld_loss = 0.0
        num_layers = len(latent_embeddings)
        
        for layer_idx in range(num_layers):
            z = latent_embeddings[layer_idx]  # [batch_size, seq_len, d_model]
            A = attention_weights[layer_idx]  # [batch_size, nhead, seq_len, seq_len]
            
            # Average over attention heads
            A_avg = A.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Calculate GraphSim for all token pairs
            graph_sim = torch.zeros_like(target_sim)
            
            for i in range(seq_len):
                for j in range(seq_len):
                    # Skip if attention weight is very small
                    if A_avg[:, i, j].mean() < 1e-5:
                        continue
                    
                    # z_i: [batch_size, d_model]
                    # A_ij: [batch_size]
                    # z_j: [batch_size, d_model]
                    z_i = z[:, i]
                    a_ij = A_avg[:, i, j].unsqueeze(-1)
                    z_j = z[:, j]
                    
                    # Calculate attended vector: A_ij * z_j
                    attended_j = a_ij * z_j
                    
                    # Calculate cosine similarity
                    sim_ij = F.cosine_similarity(z_i, attended_j, dim=1)
                    
                    # Add to graph similarity matrix
                    graph_sim[:, i] += sim_ij
            
            # Normalize by number of connections
            graph_sim = graph_sim / seq_len
            
            # Calculate RLD loss as difference between GraphSim and TargetSim
            layer_rld = torch.abs(graph_sim - target_sim).mean()
            rld_loss += layer_rld
        
        # Average over layers
        rld_loss = rld_loss / num_layers
        return rld_loss
    
    def _compute_cel_loss(self, attention_weights: list) -> torch.Tensor:
        """
        Compute Computational Entropy Loss (CEL).
        
        CEL = -(1/T) ∑_{t=1}^T E_{p(A_t)}[log p(A_t) · (1/eff(A_t))]
        """
        cel_loss = 0.0
        num_layers = len(attention_weights)
        
        for layer_idx in range(num_layers):
            A = attention_weights[layer_idx]  # [batch_size, nhead, seq_len, seq_len]
            batch_size, nhead, seq_len, _ = A.shape
            
            # Process each attention head separately
            for head_idx in range(nhead):
                # Get attention probabilities for current head
                A_head = A[:, head_idx]  # [batch_size, seq_len, seq_len]
                
                # Calculate attention entropy
                # Add a small epsilon to avoid log(0)
                entropy = -torch.sum(A_head * torch.log(A_head + self.eps), dim=-1)  # [batch_size, seq_len]
                
                # Calculate efficiency measure
                variance = torch.var(A_head, dim=-1)  # [batch_size, seq_len]
                
                # Calculate kurtosis
                # Kurtosis = E[(X - μ)^4] / (E[(X - μ)^2])^2 - 3
                mean = torch.mean(A_head, dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
                diff = A_head - mean  # [batch_size, seq_len, seq_len]
                diff_sq = diff ** 2  # [batch_size, seq_len, seq_len]
                var = torch.mean(diff_sq, dim=-1)  # [batch_size, seq_len]
                kurtosis = torch.mean(diff ** 4, dim=-1) / (var ** 2 + self.eps) - 3  # [batch_size, seq_len]
                
                # Calculate efficiency
                eff = variance + self.kappa * kurtosis / self.c_const  # [batch_size, seq_len]
                eff = torch.clamp(eff, min=self.eps)  # Ensure positive values
                
                # Calculate CEL for current head
                head_cel = torch.mean(entropy / eff)
                
                cel_loss += head_cel
        
        # Average over layers and heads
        cel_loss = cel_loss / (num_layers * nhead)
        return cel_loss
    
    def _compute_cfr_loss(self, layer_losses: list) -> torch.Tensor:
        """
        Compute Causal Feedback Reinforcement (CFR) loss.
        
        CFR = (1/(L-1)) ∑_{l=1}^{L-1} |∂L_{l+1}/∂L_l · c_{l,l+1}|
        
        where c_{l,l+1} is the regression-based causal coefficient.
        """
        L = len(layer_losses)
        if L <= 1:
            return torch.tensor(0.0, device=layer_losses[0].device)
        
        # Compute gradients between adjacent layers
        cfr_loss = 0.0
        for l in range(L - 1):
            # Compute gradient sensitivity
            L_l = layer_losses[l]
            L_l1 = layer_losses[l + 1]
            
            # Compute gradient of L_{l+1} with respect to L_l
            # We can approximate this using autograd
            L_l.retain_grad()
            L_l1.backward(retain_graph=True)
            grad = L_l.grad if L_l.grad is not None else torch.tensor(0.0, device=L_l.device)
            
            # Compute causal coefficient c_{l,l+1}
            # Using a simplification for the demonstration
            c_ll1 = 0.5  # This would be computed using regression in practice
            
            # Add to CFR loss
            cfr_loss += torch.abs(grad * c_ll1)
            
            # Clear gradients
            L_l.grad = None
        
        # Average over layers
        cfr_loss = cfr_loss / (L - 1)
        return cfr_loss


class EnhancedEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-scale sparse attention.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        sparse_rank: int = 64
    ):
        super().__init__()
        
        # Dense branch components
        self.self_attn_dense = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Sparse branch components
        self.sparse_rank = sparse_rank
        self.Q_proj = nn.Linear(d_model, sparse_rank)
        self.K_proj = nn.Linear(d_model, sparse_rank)
        self.V_proj = nn.Linear(d_model, d_model)
        
        # Branch fusion parameter (learnable)
        self.lambda_fusion = nn.Parameter(torch.tensor(0.5))
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple:
        """
        Forward pass with optional return of attention weights.
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Mask for self-attention
            src_key_padding_mask: Mask for padding tokens
            return_attention: Whether to return attention weights
            
        Returns:
            output: Processed tensor
            attn_weights: Attention weights if return_attention is True
            latent_emb: Latent embeddings if return_attention is True
        """
        # Store latent embeddings before self-attention
        latent_emb = src
        
        # Dense branch: standard multi-head attention
        src_t = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        dense_output, attn_weights_dense = self.self_attn_dense(
            src_t, src_t, src_t,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        dense_output = dense_output.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Sparse branch: low-rank approximation
        Q = self.Q_proj(src)  # [batch_size, seq_len, sparse_rank]
        K = self.K_proj(src)  # [batch_size, seq_len, sparse_rank]
        V = self.V_proj(src)  # [batch_size, seq_len, d_model]
        
        # Compute sparse attention weights
        QK = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_len, seq_len]
        QK = QK / math.sqrt(self.sparse_rank)
        
        # Apply masking if provided
        if src_mask is not None:
            QK = QK.masked_fill(src_mask == 0, float('-inf'))
        
        if src_key_padding_mask is not None:
            QK = QK.masked_fill(src_key_padding_mask.unsqueeze(1), float('-inf'))
        
        # Apply softmax to get probabilities
        attn_weights_sparse = F.softmax(QK, dim=-1)
        
        # Apply thresholding to enforce sparsity
        # Keep only top-k values per query
        k = max(1, int(attn_weights_sparse.size(-1) * 0.2))  # Keep top 20%
        topk_values, topk_indices = torch.topk(attn_weights_sparse, k, dim=-1)
        
        # Create sparse attention matrix
        sparse_attn = torch.zeros_like(attn_weights_sparse)
        for i in range(attn_weights_sparse.size(0)):
            for j in range(attn_weights_sparse.size(1)):
                sparse_attn[i, j, topk_indices[i, j]] = topk_values[i, j]
        
        # Normalize sparse attention weights
        sparse_attn = sparse_attn / (sparse_attn.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Apply attention to values
        sparse_output = torch.bmm(sparse_attn, V)  # [batch_size, seq_len, d_model]
        
        # Fuse dense and sparse branches
        lambda_sigmoid = torch.sigmoid(self.lambda_fusion)
        fused_output = lambda_sigmoid * dense_output + (1 - lambda_sigmoid) * sparse_output
        
        # Add residual connection and apply normalization
        src = src + self.dropout(fused_output)
        src = self.norm1(src)
        
        # Feed-forward network
        ff_output = self.feed_forward(src)
        
        # Add residual connection and apply normalization
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        if return_attention:
            # Combine attention weights from both branches
            combined_attn = lambda_sigmoid * attn_weights_dense + (1 - lambda_sigmoid) * sparse_attn
            return src, combined_attn, latent_emb
        else:
            return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 1) Build a [max_len × d_model] matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 2) Unsqueeze to shape [1 × max_len × d_model]
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MetaLearner(nn.Module):
    """
    Meta-learner for dynamic loss weighting.
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 32):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        self.c_scale = 2.0  # Scale factor for normalization
    
    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights for loss components.
        
        Args:
            loss_values: Tensor of loss values [input_dim]
            
        Returns:
            Normalized weights for each loss component
        """
        # Pass through the network
        weights = F.softplus(self.model(loss_values))
        
        # Normalize weights to sum to 1
        weights_sum = weights.sum() + 1e-8
        normalized_weights = weights / weights_sum
        
        # Scale by constant factor
        scaled_weights = normalized_weights * self.c_scale
        
        return scaled_weights


# Utility functions for training and evaluation

def train_epoch(
    model: EnhancedEURLTransformer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Dict:
    """Training loop for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {
        'ce_loss': 0, 'rld_loss': 0, 'cel_loss': 0, 'cfr_loss': 0,
        'alpha': 0, 'beta': 0, 'gamma': 0
    }
    
    for batch in dataloader:
        # Get data
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        
        # Forward pass with internal states
        logits, internal_states = model(src, return_internal_states=True)
        
        # Compute loss
        loss, batch_loss_components = model.compute_loss(logits, tgt, internal_states)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        for k, v in batch_loss_components.items():
            loss_components[k] += v
    
    # Compute averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return {'loss': avg_loss, **loss_components}


def evaluate(
    model: EnhancedEURLTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict:
    """Evaluation loop."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get data
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass
            logits, internal_states = model(src, return_internal_states=True)
            
            # Compute loss
            loss, _ = model.compute_loss(logits, tgt, internal_states)
            
            # Compute accuracy
            pred = logits.argmax(dim=-1)
            correct = (pred == tgt).sum().item()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_tokens += tgt.numel()
    
    # Compute averages
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    
    return {'loss': avg_loss, 'accuracy': accuracy}


# Example usage
def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EnhancedEURLTransformer(
        vocab_size=30000,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        sparse_rank=64
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Train model (with proper data loaders)
    # train_epoch(model, train_dataloader, optimizer, device)
    
    # Evaluate model
    # eval_metrics = evaluate(model, val_dataloader, device)
    
    print("Enhanced EURL model ready!")


if __name__ == "__main__":
    main()