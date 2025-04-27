import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import os
import json
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# Import EURL model (assume the implementation exists in eurl_model.py)
from eurl_model import EnhancedEURLTransformer

# Import baseline models for comparison
from transformers import BertModel, BertConfig, AutoTokenizer, AutoModel


class BenchmarkingFramework:
    """Framework for benchmarking EURL against baseline models."""
    
    def __init__(
        self,
        dataset_name: str,
        task_type: str,
        model_config: Dict,
        output_dir: str,
        device: torch.device = None
    ):
        """
        Initialize the benchmarking framework.
        
        Args:
            dataset_name: Name of the dataset ('wikitext-103', 'wmt14', 'mscoco')
            task_type: Type of task ('lm' for language modeling, 'mt' for machine translation, 'ic' for image captioning)
            model_config: Configuration for models
            output_dir: Directory to save results
            device: Device to run on (default: cuda if available, otherwise cpu)
        """
        self.dataset_name = dataset_name
        self.task_type = task_type
        self.model_config = model_config
        self.output_dir = output_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'eurl': self._create_eurl_model(),
            'baseline': self._create_baseline_model(),
            'sparse_transformer': self._create_sparse_transformer()
        }
        
        # Initialize metrics
        self.metrics = {
            'training_time': {},
            'convergence_epochs': {},
            'final_performance': {},
            'attention_entropy': {},
            'latent_attention_alignment': {},
            'interlayer_alignment': {}
        }
        
        # Load dataset
        self.train_loader, self.val_loader, self.test_loader = self._load_dataset()
    
    def _create_eurl_model(self) -> EnhancedEURLTransformer:
        """Create the Enhanced EURL model."""
        return EnhancedEURLTransformer(
            vocab_size=self.model_config['vocab_size'],
            d_model=self.model_config['d_model'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            dropout=self.model_config['dropout'],
            sparse_rank=self.model_config['sparse_rank'],
            kappa=self.model_config.get('kappa', 0.5),
            c_const=self.model_config.get('c_const', 2.0)
        ).to(self.device)
    
    def _create_baseline_model(self) -> nn.Module:
        """Create the baseline Transformer model."""
        if self.task_type == 'lm':
            # For language modeling, use a standard Transformer
            config = BertConfig(
                vocab_size=self.model_config['vocab_size'],
                hidden_size=self.model_config['d_model'],
                num_hidden_layers=self.model_config['num_layers'],
                num_attention_heads=self.model_config['nhead'],
                intermediate_size=self.model_config['dim_feedforward'],
                hidden_dropout_prob=self.model_config['dropout']
            )
            model = BertModel(config)
            # Add LM head
            model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size)
            return model.to(self.device)
        elif self.task_type == 'mt':
            # Use a pre-trained model for machine translation
            model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
            return model.to(self.device)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _create_sparse_transformer(self) -> nn.Module:
        """Create a sparse transformer model for comparison."""
        # Implementation of Sparse Transformer (Child et al., 2019)
        # This is a simplified version for comparison
        return EnhancedEURLTransformer(
            vocab_size=self.model_config['vocab_size'],
            d_model=self.model_config['d_model'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            dropout=self.model_config['dropout'],
            # Only use sparse attention, no meta-learner or additional losses
            sparse_rank=self.model_config['sparse_rank']
        ).to(self.device)
    
    def _load_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load dataset based on the task type."""
        # Placeholder for dataset loading
        # In a real implementation, you would load the actual datasets
        if self.dataset_name == 'wikitext-103':
            # Return WikiText-103 data loaders
            pass
        elif self.dataset_name == 'wmt14':
            # Return WMT-14 data loaders
            pass
        elif self.dataset_name == 'mscoco':
            # Return MSCOCO data loaders
            pass
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        # For demonstration purposes, return dummy data loaders
        dummy_loader = DataLoader(torch.utils.data.TensorDataset(
            torch.randint(0, self.model_config['vocab_size'], (100, 50)),  # Input tokens
            torch.randint(0, self.model_config['vocab_size'], (100, 50))   # Target tokens
        ), batch_size=16)
        
        return dummy_loader, dummy_loader, dummy_loader
    
    def train_model(self, model_name: str, num_epochs: int, lr: float = 1e-4) -> Dict:
        """
        Train a specific model and collect metrics.
        
        Args:
            model_name: Name of the model to train ('eurl', 'baseline', 'sparse_transformer')
            num_epochs: Number of epochs to train
            lr: Learning rate
            
        Returns:
            Dictionary of training metrics
        """
        model = self.models[model_name]
        model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Setup metrics collection
        metrics = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'attention_entropies': [],
            'latent_alignments': [],
            'interlayer_alignments': []
        }
        
        # Define convergence threshold
        convergence_threshold = 0.001
        converged_epoch = num_epochs  # Default to max epochs
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss = self._train_epoch(model, optimizer, model_name)
            metrics['train_losses'].append(train_loss)
            
            # Evaluate on validation set
            val_metrics = self._evaluate(model, self.val_loader, model_name)
            metrics['val_losses'].append(val_metrics['loss'])
            metrics['val_accuracies'].append(val_metrics.get('accuracy', 0))
            
            # Collect model-specific metrics
            if model_name == 'eurl':
                attn_entropy = self._measure_attention_entropy(model)
                latent_align = self._measure_latent_alignment(model)
                interlayer_align = self._measure_interlayer_alignment(model)
                
                metrics['attention_entropies'].append(attn_entropy)
                metrics['latent_alignments'].append(latent_align)
                metrics['interlayer_alignments'].append(interlayer_align)
            
            # Check for convergence
            if epoch > 0 and abs(metrics['val_losses'][-1] - metrics['val_losses'][-2]) < convergence_threshold:
                converged_epoch = epoch + 1
                print(f"Model {model_name} converged at epoch {converged_epoch}")
                break
            
            # Record epoch time
            epoch_time = time.time() - epoch_start
            metrics['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {metrics['val_losses'][-1]:.4f}, Time: {epoch_time:.2f}s")
        
        # Record total training time
        total_time = time.time() - start_time
        
        # Store metrics
        self.metrics['training_time'][model_name] = total_time
        self.metrics['convergence_epochs'][model_name] = converged_epoch
        self.metrics['final_performance'][model_name] = {
            'val_loss': metrics['val_losses'][-1],
            'val_accuracy': metrics['val_accuracies'][-1]
        }
        
        if model_name == 'eurl':
            self.metrics['attention_entropy'][model_name] = metrics['attention_entropies'][-1]
            self.metrics['latent_attention_alignment'][model_name] = metrics['latent_alignments'][-1]
            self.metrics['interlayer_alignment'][model_name] = metrics['interlayer_alignments'][-1]
        
        return metrics
    
    def _train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, model_name: str) -> float:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            model_name: Name of the model
            
        Returns:
            Average training loss
        """
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc=f"Training {model_name}"):
            # Get data
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if model_name == 'eurl':
                logits, internal_states = model(src, return_internal_states=True)
                loss, _ = model.compute_loss(logits, tgt, internal_states)
            else:
                # Simplified loss computation for other models
                if hasattr(model, 'lm_head'):
                    # For language modeling
                    outputs = model(src)[0]  # Get last hidden states
                    logits = model.lm_head(outputs)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
                else:
                    # Generic approach
                    logits = model(src)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _evaluate(self, model: nn.Module, dataloader: DataLoader, model_name: str) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Forward pass
                if model_name == 'eurl':
                    logits, internal_states = model(src, return_internal_states=True)
                    loss, _ = model.compute_loss(logits, tgt, internal_states)
                else:
                    # Simplified loss computation for other models
                    if hasattr(model, 'lm_head'):
                        # For language modeling
                        outputs = model(src)[0]  # Get last hidden states
                        logits = model.lm_head(outputs)
                    else:
                        # Generic approach
                        logits = model(src)
                    
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
                
                # Compute accuracy
                pred = logits.argmax(dim=-1)
                correct = (pred == tgt).sum().item()
                
                # Update metrics
                total_loss += loss.item()
                total_correct += correct
                total_tokens += tgt.numel()
        
        # Compute averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_tokens
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _measure_attention_entropy(self, model: EnhancedEURLTransformer) -> float:
        """
        Measure attention entropy for EURL model.
        
        Args:
            model: EURL model
            
        Returns:
            Average attention entropy across layers and heads
        """
        model.eval()
        total_entropy = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src, _ = batch
                src = src.to(self.device)
                
                # Forward pass to get attention weights
                _, internal_states = model(src, return_internal_states=True)
                attention_weights = internal_states['attention_weights']
                
                # Calculate entropy for each layer and head
                for layer_attn in attention_weights:
                    batch_size, nhead, seq_len, _ = layer_attn.shape
                    
                    for head_idx in range(nhead):
                        # Get attention probabilities for current head
                        A_head = layer_attn[:, head_idx]  # [batch_size, seq_len, seq_len]
                        
                        # Calculate attention entropy
                        # Add a small epsilon to avoid log(0)
                        entropy = -torch.sum(A_head * torch.log(A_head + 1e-8), dim=-1)
                        
                        # Average over batch and sequence length
                        total_entropy += entropy.mean().item()
                        num_samples += 1
        
        return total_entropy / num_samples if num_samples > 0 else 0
    
    def _measure_latent_alignment(self, model: EnhancedEURLTransformer) -> float:
        """
        Measure latent-attention alignment for EURL model.
        
        Args:
            model: EURL model
            
        Returns:
            Average cosine similarity between latent embeddings and attention-weighted vectors
        """
        model.eval()
        total_alignment = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src, _ = batch
                src = src.to(self.device)
                
                # Forward pass to get attention weights and latent embeddings
                _, internal_states = model(src, return_internal_states=True)
                attention_weights = internal_states['attention_weights']
                latent_embeddings = internal_states['latent_embeddings']
                
                # Calculate alignment for each layer
                for layer_idx in range(len(latent_embeddings)):
                    z = latent_embeddings[layer_idx]  # [batch_size, seq_len, d_model]
                    A = attention_weights[layer_idx]  # [batch_size, nhead, seq_len, seq_len]
                    
                    # Average over attention heads
                    A_avg = A.mean(dim=1)  # [batch_size, seq_len, seq_len]
                    
                    # Calculate average alignment score
                    batch_size, seq_len, _ = z.shape
                    alignment_scores = torch.zeros(batch_size, seq_len, device=self.device)
                    
                    for i in range(seq_len):
                        # Calculate attended vectors: sum_j(A_ij * z_j)
                        attended_vectors = torch.bmm(
                            A_avg[:, i:i+1, :],  # [batch_size, 1, seq_len]
                            z  # [batch_size, seq_len, d_model]
                        )  # [batch_size, 1, d_model]
                        
                        # Calculate cosine similarity
                        z_i = z[:, i:i+1, :]  # [batch_size, 1, d_model]
                        sim = F.cosine_similarity(z_i, attended_vectors, dim=2)
                        alignment_scores[:, i] = sim.squeeze()
                    
                    # Average over batch and sequence length
                    total_alignment += alignment_scores.mean().item()
                    num_samples += 1
        
        return total_alignment / num_samples if num_samples > 0 else 0
    
    def _measure_interlayer_alignment(self, model: EnhancedEURLTransformer) -> float:
        """
        Measure inter-layer alignment for EURL model.
        
        Args:
            model: EURL model
            
        Returns:
            Average alignment coefficient between adjacent layers
        """
        model.eval()
        total_alignment = 0
        num_pairs = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                src, _ = batch
                src = src.to(self.device)
                
                # Forward pass to get layer outputs
                _, internal_states = model(src, return_internal_states=True)
                layer_outputs = internal_states['layer_outputs']
                
                # Calculate alignment between adjacent layers
                for l in range(len(layer_outputs) - 1):
                    out_l = layer_outputs[l]  # [batch_size, seq_len, d_model]
                    out_l1 = layer_outputs[l + 1]  # [batch_size, seq_len, d_model]
                    
                    # Calculate correlation coefficient
                    # Flatten the outputs
                    out_l_flat = out_l.reshape(out_l.size(0), -1)  # [batch_size, seq_len * d_model]
                    out_l1_flat = out_l1.reshape(out_l1.size(0), -1)  # [batch_size, seq_len * d_model]
                    
                    # Calculate mean and variance
                    mean_l = out_l_flat.mean(dim=1, keepdim=True)
                    mean_l1 = out_l1_flat.mean(dim=1, keepdim=True)
                    
                    var_l = ((out_l_flat - mean_l) ** 2).mean(dim=1)
                    var_l1 = ((out_l1_flat - mean_l1) ** 2).mean(dim=1)
                    
                    # Calculate covariance
                    cov = ((out_l_flat - mean_l) * (out_l1_flat - mean_l1)).mean(dim=1)
                    
                    # Calculate correlation coefficient
                    corr = cov / (torch.sqrt(var_l * var_l1) + 1e-8)
                    
                    # Average over batch
                    total_alignment += corr.mean().item()
                    num_pairs += 1
        
        return total_alignment / num_pairs if num_pairs > 0 else 0
    
    def run_full_comparison(self, num_epochs: int = 10, lr: float = 1e-4) -> Dict:
        """
        Run a full comparison of all models.
        
        Args:
            num_epochs: Number of epochs to train
            lr: Learning rate
            
        Returns:
            Compiled metrics for all models
        """
        print(f"Running full comparison on {self.dataset_name} for {self.task_type} task")
        
        # Train and evaluate each model
        for model_name in self.models:
            print(f"\nTraining {model_name} model...")
            metrics = self.train_model(model_name, num_epochs, lr)
            
            # Save training curves
            self._plot_training_curves(model_name, metrics)
        
        # Compile comparison results
        comparison = self._compile_comparison_results()
        
        # Save results to file
        results_path = os.path.join(self.output_dir, f"comparison_results_{self.dataset_name}.json")
        with open(results_path, 'w') as f:
            json.dump(comparison, f, indent=4)
        
        # Create comparison visualizations
        self._plot_comparison_results(comparison)
        
        return comparison
    
    def _plot_training_curves(self, model_name: str, metrics: Dict) -> None:
        """
        Plot training curves for a model.
        
        Args:
            model_name: Name of the model
            metrics: Training metrics
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        epochs = range(1, len(metrics['train_losses']) + 1)
        axs[0, 0].plot(epochs, metrics['train_losses'], 'b-', label='Train Loss')
        axs[0, 0].plot(epochs, metrics['val_losses'], 'r-', label='Val Loss')
        axs[0, 0].set_title(f"{model_name} - Loss Curves")
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        
        # Plot validation accuracy
        axs[0, 1].plot(epochs, metrics['val_accuracies'], 'g-')
        axs[0, 1].set_title(f"{model_name} - Validation Accuracy")
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        
        # Plot epoch times
        axs[1, 0].plot(epochs, metrics['epoch_times'], 'm-')
        axs[1, 0].set_title(f"{model_name} - Epoch Times")
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Time (s)')
        
        # Plot model-specific metrics if available
        if model_name == 'eurl' and 'attention_entropies' in metrics:
            ax2 = axs[1, 1]
            
            # Plot on primary y-axis
            color1, color2, color3 = 'tab:blue', 'tab:orange', 'tab:green'
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Attention Entropy', color=color1)
            line1, = ax2.plot(epochs, metrics['attention_entropies'], color=color1, label='Attn Entropy')
            ax2.tick_params(axis='y', labelcolor=color1)
            
            # Create secondary y-axis
            ax3 = ax2.twinx()
            ax3.set_ylabel('Alignment', color=color2)
            line2, = ax3.plot(epochs, metrics['latent_alignments'], color=color2, label='Latent Align')
            line3, = ax3.plot(epochs, metrics['interlayer_alignments'], color=color3, label='Layer Align')
            ax3.tick_params(axis='y', labelcolor=color2)
            
            # Add legend
            lines = [line1, line2, line3]
            ax2.legend(lines, [l.get_label() for l in lines])
            
            ax2.set_title(f"{model_name} - Specialized Metrics")
        else:
            # Empty plot for non-EURL models
            axs[1, 1].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_training_curves.png"))
        plt.close()
    
    def _compile_comparison_results(self) -> Dict:
        """
        Compile comparison results for all models.
        
        Returns:
            Dictionary of comparison results
        """
        comparison = {
            'dataset': self.dataset_name,
            'task': self.task_type,
            'models': {},
            'comparative_metrics': {
                'training_time_speedup': {},
                'convergence_speedup': {},
                'performance_improvement': {}
            }
        }
        
        # Add data for each model
        for model_name in self.models:
            comparison['models'][model_name] = {
                'training_time': self.metrics['training_time'].get(model_name, 0),
                'convergence_epochs': self.metrics['convergence_epochs'].get(model_name, 0),
                'final_performance': self.metrics['final_performance'].get(model_name, {})
            }
            
            # Add model-specific metrics for EURL
            if model_name == 'eurl':
                comparison['models'][model_name].update({
                    'attention_entropy': self.metrics['attention_entropy'].get(model_name, 0),
                    'latent_attention_alignment': self.metrics['latent_attention_alignment'].get(model_name, 0),
                    'interlayer_alignment': self.metrics['interlayer_alignment'].get(model_name, 0)
                })
        
        # Calculate comparative metrics
        if 'baseline' in self.models and 'eurl' in self.models:
            # Training time speedup
            baseline_time = comparison['models']['baseline']['training_time']
            eurl_time = comparison['models']['eurl']['training_time']
            if baseline_time > 0:
                comparison['comparative_metrics']['training_time_speedup']['vs_baseline'] = f"{baseline_time / eurl_time:.2f}x"
            
            # Convergence speedup
            baseline_epochs = comparison['models']['baseline']['convergence_epochs']
            eurl_epochs = comparison['models']['eurl']['convergence_epochs']
            if baseline_epochs > 0:
                comparison['comparative_metrics']['convergence_speedup']['vs_baseline'] = f"{baseline_epochs / eurl_epochs:.2f}x"
            
            # Performance improvement
            baseline_perf = comparison['models']['baseline']['final_performance'].get('val_accuracy', 0)
            eurl_perf = comparison['models']['eurl']['final_performance'].get('val_accuracy', 0)
            if baseline_perf > 0:
                comparison['comparative_metrics']['performance_improvement']['vs_baseline'] = f"{(eurl_perf - baseline_perf) / baseline_perf * 100:.2f}%"
        
        if 'sparse_transformer' in self.models and 'eurl' in self.models:
            # Training time speedup
            sparse_time = comparison['models']['sparse_transformer']['training_time']
            eurl_time = comparison['models']['eurl']['training_time']
            if sparse_time > 0:
                comparison['comparative_metrics']['training_time_speedup']['vs_sparse'] = f"{sparse_time / eurl_time:.2f}x"
            
            # Convergence speedup
            sparse_epochs = comparison['models']['sparse_transformer']['convergence_epochs']
            eurl_epochs = comparison['models']['eurl']['convergence_epochs']
            if sparse_epochs > 0:
                comparison['comparative_metrics']['convergence_speedup']['vs_sparse'] = f"{sparse_epochs / eurl_epochs:.2f}x"
            
            # Performance improvement
            sparse_perf = comparison['models']['sparse_transformer']['final_performance'].get('val_accuracy', 0)
            eurl_perf = comparison['models']['eurl']['final_performance'].get('val_accuracy', 0)
            if sparse_perf > 0:
                comparison['comparative_metrics']['performance_improvement']['vs_sparse'] = f"{(eurl_perf - sparse_perf) / sparse_perf * 100:.2f}%"
        
        return comparison
    
    def _plot_comparison_results(self, comparison: Dict) -> None:
        """
        Plot comparison results.
        
        Args:
            comparison: Comparison results
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get model names
        model_names = list(comparison['models'].keys())
        
        # Training time comparison
        training_times = [comparison['models'][model]['training_time'] for model in model_names]
        axs[0, 0].bar(model_names, training_times)
        axs[0, 0].set_title('Training Time Comparison')
        axs[0, 0].set_ylabel('Time (s)')
        
        # Convergence speed comparison
        convergence_epochs = [comparison['models'][model]['convergence_epochs'] for model in model_names]
        axs[0, 1].bar(model_names, convergence_epochs)
        axs[0, 1].set_title('Convergence Speed Comparison')
        axs[0, 1].set_ylabel('Epochs')
        
        # Performance comparison
        performances = [comparison['models'][model]['final_performance'].get('val_accuracy', 0) for model in model_names]
        axs[1, 0].bar(model_names, performances)
        axs[1, 0].set_title('Performance Comparison')
        axs[1, 0].set_ylabel('Validation Accuracy')
        
        # EURL-specific metrics
        if 'eurl' in model_names:
            eurl_metrics = {
                'Attention Entropy': comparison['models']['eurl'].get('attention_entropy', 0),
                'Latent Alignment': comparison['models']['eurl'].get('latent_attention_alignment', 0),
                'Interlayer Alignment': comparison['models']['eurl'].get('interlayer_alignment', 0)
            }
            
            metric_names = list(eurl_metrics.keys())
            metric_values = list(eurl_metrics.values())
            
            axs[1, 1].bar(metric_names, metric_values)
            axs[1, 1].set_title('EURL Specialized Metrics')
            axs[1, 1].set_ylabel('Value')
        else:
            axs[1, 1].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"model_comparison_{self.dataset_name}.png"))
        plt.close()


# Example usage
def run_benchmarks():
    model_config = {
        'vocab_size': 30000,
        'd_model': 512,
        'nhead': 8,
        'num_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'sparse_rank': 64,
        'kappa': 0.5,
        'c_const': 2.0
    }
    
    output_dir = "./benchmark_results"
    
    # Run benchmarks for language modeling on WikiText-103
    wikitext_benchmark = BenchmarkingFramework(
        dataset_name="wikitext-103",
        task_type="lm",
        model_config=model_config,
        output_dir=os.path.join(output_dir, "wikitext-103")
    )
    wikitext_results = wikitext_benchmark.run_full_comparison(num_epochs=10)
    
    # Run benchmarks for machine translation on WMT-14
    wmt_benchmark = BenchmarkingFramework(
        dataset_name="wmt14",
        task_type="mt",
        model_config=model_config,
        output_dir=os.path.join(output_dir, "wmt14")
    )
    wmt_results = wmt_benchmark.run_full_comparison(num_epochs=10)
    
    # Compile overall results
    all_results = {
        "wikitext-103": wikitext_results,
        "wmt14": wmt_results
    }
    
    # Save overall results
    with open(os.path.join(output_dir, "overall_benchmark_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("Benchmarking complete. Results saved to:", output_dir)


if __name__ == "__main__":
    run_benchmarks()