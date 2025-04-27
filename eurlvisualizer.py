import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

class EURLVisualizer:
    """
    Visualization toolkit for analyzing EURL model behavior and performance.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up color maps
        self.attention_cmap = plt.cm.viridis
        self.diff_cmap = self._create_diverging_colormap()
    
    def _create_diverging_colormap(self) -> LinearSegmentedColormap:
        """Create a custom diverging colormap for difference visualization."""
        return LinearSegmentedColormap.from_list(
            "custom_diverging",
            ["#2c7bb6", "#ffffbf", "#d7191c"],
            N=256
        )
    
    def visualize_attention_patterns(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        sentence_idx: int = 0,
        title: str = "Attention Pattern",
        filename: str = "attention_pattern.png"
    ) -> None:
        """
        Visualize attention patterns from a specific layer and head.
        
        Args:
            attention_weights: Attention weights tensor [batch_size, num_heads, seq_len, seq_len]
            tokens: List of token strings (optional)
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            sentence_idx: Index of the sentence in the batch to visualize
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Extract attention weights for the specified layer, head, and sentence
        attn = attention_weights[sentence_idx, head_idx].cpu().detach().numpy()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot attention matrix
        sns.heatmap(
            attn,
            annot=False,
            cmap=self.attention_cmap,
            xticklabels=tokens if tokens else [],
            yticklabels=tokens if tokens else [],
            cbar=True
        )
        
        # Set title and labels
        plt.title(f"{title} (Layer {layer_idx}, Head {head_idx})")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        
        # Rotate x-axis labels for better readability
        if tokens:
            plt.xticks(rotation=45, ha="right")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_attention_patterns(
        self,
        base_attention: torch.Tensor,
        eurl_attention: torch.Tensor,
        tokens: List[str] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        sentence_idx: int = 0,
        title: str = "Attention Pattern Comparison",
        filename: str = "attention_comparison.png"
    ) -> None:
        """
        Compare attention patterns between baseline and EURL models.
        
        Args:
            base_attention: Baseline attention weights tensor
            eurl_attention: EURL attention weights tensor
            tokens: List of token strings (optional)
            layer_idx: Index of the layer to visualize
            head_idx: Index of the attention head to visualize
            sentence_idx: Index of the sentence in the batch to visualize
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Extract attention weights for the specified layer, head, and sentence
        base_attn = base_attention[sentence_idx, head_idx].cpu().detach().numpy()
        eurl_attn = eurl_attention[sentence_idx, head_idx].cpu().detach().numpy()
        
        # Calculate difference
        diff_attn = eurl_attn - base_attn
        
        # Create figure with subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot baseline attention
        sns.heatmap(
            base_attn,
            annot=False,
            cmap=self.attention_cmap,
            xticklabels=tokens if tokens else [],
            yticklabels=tokens if tokens else [],
            cbar=True,
            ax=axs[0]
        )
        axs[0].set_title(f"Baseline Attention")
        axs[0].set_xlabel("Key Tokens")
        axs[0].set_ylabel("Query Tokens")
        
        # Plot EURL attention
        sns.heatmap(
            eurl_attn,
            annot=False,
            cmap=self.attention_cmap,
            xticklabels=tokens if tokens else [],
            yticklabels=tokens if tokens else [],
            cbar=True,
            ax=axs[1]
        )
        axs[1].set_title(f"EURL Attention")
        axs[1].set_xlabel("Key Tokens")
        axs[1].set_ylabel("Query Tokens")
        
        # Plot difference
        vmax = max(abs(diff_attn.min()), abs(diff_attn.max()))
        sns.heatmap(
            diff_attn,
            annot=False,
            cmap=self.diff_cmap,
            xticklabels=tokens if tokens else [],
            yticklabels=tokens if tokens else [],
            cbar=True,
            vmin=-vmax,
            vmax=vmax,
            ax=axs[2]
        )
        axs[2].set_title(f"Difference (EURL - Baseline)")
        axs[2].set_xlabel("Key Tokens")
        axs[2].set_ylabel("Query Tokens")
        
        # Rotate x-axis labels for better readability
        if tokens:
            for ax in axs:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        title: str = "Attention Entropy Distribution",
        filename: str = "attention_entropy.png"
    ) -> None:
        """
        Visualize the entropy distribution of attention weights.
        
        Args:
            attention_weights: Attention weights tensor [batch_size, num_heads, seq_len, seq_len]
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Calculate entropy for each attention distribution
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + epsilon),
            dim=-1
        ).cpu().detach().numpy()  # [batch_size, num_heads, seq_len]
        
        # Flatten entropy values
        entropy_flat = entropy.flatten()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot entropy distribution
        sns.histplot(entropy_flat, kde=True)
        
        # Set title and labels
        plt.title(title)
        plt.xlabel("Entropy (bits)")
        plt.ylabel("Frequency")
        
        # Add vertical line for mean entropy
        mean_entropy = np.mean(entropy_flat)
        plt.axvline(mean_entropy, color='r', linestyle='--')
        plt.text(
            mean_entropy + 0.1,
            plt.ylim()[1] * 0.9,
            f"Mean: {mean_entropy:.2f} bits",
            color='r'
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_attention_entropy(
        self,
        base_attention: torch.Tensor,
        eurl_attention: torch.Tensor,
        title: str = "Attention Entropy Comparison",
        filename: str = "entropy_comparison.png"
    ) -> None:
        """
        Compare attention entropy distributions between baseline and EURL models.
        
        Args:
            base_attention: Baseline attention weights tensor
            eurl_attention: EURL attention weights tensor
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Calculate entropy for each attention distribution
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        
        base_entropy = -torch.sum(
            base_attention * torch.log(base_attention + epsilon),
            dim=-1
        ).cpu().detach().numpy().flatten()
        
        eurl_entropy = -torch.sum(
            eurl_attention * torch.log(eurl_attention + epsilon),
            dim=-1
        ).cpu().detach().numpy().flatten()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot entropy distributions
        sns.histplot(base_entropy, color='blue', alpha=0.5, label='Baseline', kde=True)
        sns.histplot(eurl_entropy, color='orange', alpha=0.5, label='EURL', kde=True)
        
        # Set title and labels
        plt.title(title)
        plt.xlabel("Entropy (bits)")
        plt.ylabel("Frequency")
        
        # Add vertical lines for mean entropy
        base_mean = np.mean(base_entropy)
        eurl_mean = np.mean(eurl_entropy)
        
        plt.axvline(base_mean, color='blue', linestyle='--')
        plt.text(
            base_mean + 0.1,
            plt.ylim()[1] * 0.9,
            f"Base Mean: {base_mean:.2f}",
            color='blue'
        )
        
        plt.axvline(eurl_mean, color='orange', linestyle='--')
        plt.text(
            eurl_mean + 0.1,
            plt.ylim()[1] * 0.8,
            f"EURL Mean: {eurl_mean:.2f}",
            color='orange'
        )
        
        # Add improvement percentage
        improvement = (base_mean - eurl_mean) / base_mean * 100
        plt.text(
            plt.xlim()[0] + 0.5,
            plt.ylim()[1] * 0.7,
            f"Improvement: {improvement:.1f}%",
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Add legend
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_latent_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        title: str = "Latent Space Visualization",
        filename: str = "latent_space.png",
        method: str = "tsne"
    ) -> None:
        """
        Visualize the latent space using dimensionality reduction.
        
        Args:
            embeddings: Embedding tensor [num_samples, embedding_dim]
            labels: List of label strings for coloring points (optional)
            title: Title for the plot
            filename: Filename to save the visualization
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        # Convert embeddings to numpy array
        embeddings_np = embeddings.cpu().detach().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings_np)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot embeddings
        if labels is not None:
            # Get unique labels
            unique_labels = sorted(set(labels))
            
            # Create color map
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            
            # Plot each label group
            for label in unique_labels:
                mask = [l == label for l in labels]
                plt.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    c=[color_map[label]],
                    label=label,
                    alpha=0.7
                )
            
            plt.legend()
        else:
            plt.scatter(
                reduced_embeddings[:, 0],
                reduced_embeddings[:, 1],
                alpha=0.7
            )
        
        # Set title and labels
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_latent_spaces(
        self,
        base_embeddings: torch.Tensor,
        eurl_embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        title: str = "Latent Space Comparison",
        filename: str = "latent_space_comparison.png",
        method: str = "tsne"
    ) -> None:
        """
        Compare latent spaces between baseline and EURL models.
        
        Args:
            base_embeddings: Baseline embedding tensor
            eurl_embeddings: EURL embedding tensor
            labels: List of label strings for coloring points (optional)
            title: Title for the plot
            filename: Filename to save the visualization
            method: Dimensionality reduction method ('tsne' or 'pca')
        """
        # Convert embeddings to numpy arrays
        base_np = base_embeddings.cpu().detach().numpy()
        eurl_np = eurl_embeddings.cpu().detach().numpy()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        # Combine embeddings for joint reduction
        combined = np.vstack([base_np, eurl_np])
        reduced = reducer.fit_transform(combined)
        
        # Split back into baseline and EURL
        n_base = base_np.shape[0]
        reduced_base = reduced[:n_base]
        reduced_eurl = reduced[n_base:]
        
        # Create figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot baseline embeddings
        if labels is not None:
            # Get unique labels
            unique_labels = sorted(set(labels))
            
            # Create color map
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: color for label, color in zip(unique_labels, colors)}
            
            # Plot each label group for baseline
            for label in unique_labels:
                mask = [l == label for l in labels]
                axs[0].scatter(
                    reduced_base[mask, 0],
                    reduced_base[mask, 1],
                    c=[color_map[label]],
                    label=label,
                    alpha=0.7
                )
            
            # Plot each label group for EURL
            for label in unique_labels:
                mask = [l == label for l in labels]
                axs[1].scatter(
                    reduced_eurl[mask, 0],
                    reduced_eurl[mask, 1],
                    c=[color_map[label]],
                    label=label,
                    alpha=0.7
                )
            
            # Add legends
            axs[0].legend()
            axs[1].legend()
        else:
            axs[0].scatter(reduced_base[:, 0], reduced_base[:, 1], alpha=0.7)
            axs[1].scatter(reduced_eurl[:, 0], reduced_eurl[:, 1], alpha=0.7)
        
        # Set titles and labels
        axs[0].set_title(f"Baseline Latent Space")
        axs[0].set_xlabel("Dimension 1")
        axs[0].set_ylabel("Dimension 2")
        
        axs[1].set_title(f"EURL Latent Space")
        axs[1].set_xlabel("Dimension 1")
        axs[1].set_ylabel("Dimension 2")
        
        # Set overall title
        fig.suptitle(f"{title} ({method.upper()})", fontsize=16)
        
        # Save figure
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def visualize_interlayer_alignment(
        self,
        layer_outputs: List[torch.Tensor],
        title: str = "Inter-Layer Alignment",
        filename: str = "interlayer_alignment.png"
    ) -> None:
        """
        Visualize the alignment between adjacent layers.
        
        Args:
            layer_outputs: List of layer output tensors [batch_size, seq_len, hidden_dim]
            title: Title for the plot
            filename: Filename to save the visualization
        """
        if len(layer_outputs) < 2:
            print("Need at least 2 layers to visualize inter-layer alignment")
            return
        
        # Calculate correlation coefficients between adjacent layers
        correlations = []
        layer_pairs = []
        
        for i in range(len(layer_outputs) - 1):
            # Get layer outputs
            out_l = layer_outputs[i]
            out_l1 = layer_outputs[i + 1]
            
            # Flatten outputs
            batch_size = out_l.size(0)
            out_l_flat = out_l.reshape(batch_size, -1)
            out_l1_flat = out_l1.reshape(batch_size, -1)
            
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
            avg_corr = corr.mean().item()
            correlations.append(avg_corr)
            layer_pairs.append(f"{i+1}-{i+2}")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot correlation coefficients
        bars = plt.bar(layer_pairs, correlations)
        
        # Add correlation values on top of bars
        for bar, corr in zip(bars, correlations):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{corr:.3f}",
                ha='center',
                fontsize=9
            )
        
        # Set title and labels
        plt.title(title)
        plt.xlabel("Layer Pair")
        plt.ylabel("Correlation Coefficient")
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add average correlation
        avg_corr = sum(correlations) / len(correlations)
        plt.axhline(y=avg_corr, color='r', linestyle='--')
        plt.text(
            0,
            avg_corr + 0.02,
            f"Avg: {avg_corr:.3f}",
            color='r'
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_interlayer_alignment(
        self,
        base_layer_outputs: List[torch.Tensor],
        eurl_layer_outputs: List[torch.Tensor],
        title: str = "Inter-Layer Alignment Comparison",
        filename: str = "interlayer_alignment_comparison.png"
    ) -> None:
        """
        Compare inter-layer alignment between baseline and EURL models.
        
        Args:
            base_layer_outputs: List of baseline layer output tensors
            eurl_layer_outputs: List of EURL layer output tensors
            title: Title for the plot
            filename: Filename to save the visualization
        """
        if len(base_layer_outputs) < 2 or len(eurl_layer_outputs) < 2:
            print("Need at least 2 layers to visualize inter-layer alignment")
            return
        
        if len(base_layer_outputs) != len(eurl_layer_outputs):
            print("Baseline and EURL should have the same number of layers")
            return
        
        # Calculate correlation coefficients for both models
        base_correlations = []
        eurl_correlations = []
        layer_pairs = []
        
        for i in range(len(base_layer_outputs) - 1):
            # Layer pair label
            layer_pairs.append(f"{i+1}-{i+2}")
            
            # Calculate for baseline
            out_l = base_layer_outputs[i]
            out_l1 = base_layer_outputs[i + 1]
            
            # Flatten outputs
            batch_size = out_l.size(0)
            out_l_flat = out_l.reshape(batch_size, -1)
            out_l1_flat = out_l1.reshape(batch_size, -1)
            
            # Calculate correlation
            mean_l = out_l_flat.mean(dim=1, keepdim=True)
            mean_l1 = out_l1_flat.mean(dim=1, keepdim=True)
            cov = ((out_l_flat - mean_l) * (out_l1_flat - mean_l1)).mean(dim=1)
            var_l = ((out_l_flat - mean_l) ** 2).mean(dim=1)
            var_l1 = ((out_l1_flat - mean_l1) ** 2).mean(dim=1)
            corr = cov / (torch.sqrt(var_l * var_l1) + 1e-8)
            base_correlations.append(corr.mean().item())
            
            # Calculate for EURL
            out_l = eurl_layer_outputs[i]
            out_l1 = eurl_layer_outputs[i + 1]
            
            # Flatten outputs
            batch_size = out_l.size(0)
            out_l_flat = out_l.reshape(batch_size, -1)
            out_l1_flat = out_l1.reshape(batch_size, -1)
            
            # Calculate correlation
            mean_l = out_l_flat.mean(dim=1, keepdim=True)
            mean_l1 = out_l1_flat.mean(dim=1, keepdim=True)
            cov = ((out_l_flat - mean_l) * (out_l1_flat - mean_l1)).mean(dim=1)
            var_l = ((out_l_flat - mean_l) ** 2).mean(dim=1)
            var_l1 = ((out_l1_flat - mean_l1) ** 2).mean(dim=1)
            corr = cov / (torch.sqrt(var_l * var_l1) + 1e-8)
            eurl_correlations.append(corr.mean().item())
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set width of bars
        width = 0.35
        
        # Set positions of bars
        x = np.arange(len(layer_pairs))
        
        # Plot bars
        bars1 = plt.bar(x - width/2, base_correlations, width, label='Baseline', color='blue', alpha=0.7)
        bars2 = plt.bar(x + width/2, eurl_correlations, width, label='EURL', color='orange', alpha=0.7)
        
        # Add correlation values on top of bars
        for bar, corr in zip(bars1, base_correlations):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{corr:.3f}",
                ha='center',
                fontsize=9
            )
        
        for bar, corr in zip(bars2, eurl_correlations):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{corr:.3f}",
                ha='center',
                fontsize=9
            )
        
        # Set title and labels
        plt.title(title)
        plt.xlabel("Layer Pair")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(x, layer_pairs)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Add average correlations
        avg_base = sum(base_correlations) / len(base_correlations)
        avg_eurl = sum(eurl_correlations) / len(eurl_correlations)
        
        plt.axhline(y=avg_base, color='blue', linestyle='--')
        plt.axhline(y=avg_eurl, color='orange', linestyle='--')
        
        plt.text(
            0,
            avg_base + 0.02,
            f"Base Avg: {avg_base:.3f}",
            color='blue'
        )
        
        plt.text(
            0,
            avg_eurl + 0.02,
            f"EURL Avg: {avg_eurl:.3f}",
            color='orange'
        )
        
        # Add improvement percentage
        improvement = (avg_eurl - avg_base) / abs(avg_base) * 100
        plt.text(
            len(layer_pairs) - 1,
            max(avg_base, avg_eurl) + 0.05,
            f"Improvement: {improvement:.1f}%",
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Add legend
        plt.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Metrics",
        filename: str = "training_metrics.png"
    ) -> None:
        """
        Plot training metrics over epochs.
        
        Args:
            metrics: Dictionary of metrics to plot (key: metric name, value: list of values)
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Create figure
        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
        
        # Ensure axs is always a list
        if len(metrics) == 1:
            axs = [axs]
        
        # Plot each metric
        for i, (metric_name, values) in enumerate(metrics.items()):
            epochs = range(1, len(values) + 1)
            axs[i].plot(epochs, values)
            axs[i].set_title(f"{metric_name}")
            axs[i].set_ylabel(metric_name)
            
            # Add grid
            axs[i].grid(True, linestyle='--', alpha=0.7)
        
        # Set x-axis label for the bottom subplot
        axs[-1].set_xlabel("Epoch")
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def compare_training_metrics(
        self,
        base_metrics: Dict[str, List[float]],
        eurl_metrics: Dict[str, List[float]],
        metric_names: List[str],
        title: str = "Training Metrics Comparison",
        filename: str = "training_metrics_comparison.png"
    ) -> None:
        """
        Compare training metrics between baseline and EURL models.
        
        Args:
            base_metrics: Dictionary of baseline metrics
            eurl_metrics: Dictionary of EURL metrics
            metric_names: List of metric names to plot
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Create figure
        fig, axs = plt.subplots(len(metric_names), 1, figsize=(10, 4 * len(metric_names)), sharex=True)
        
        # Ensure axs is always a list
        if len(metric_names) == 1:
            axs = [axs]
        
        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            if metric_name not in base_metrics or metric_name not in eurl_metrics:
                print(f"Metric '{metric_name}' not found in both models")
                continue
                
            base_values = base_metrics[metric_name]
            eurl_values = eurl_metrics[metric_name]
            
            # Ensure same length by truncating to the shorter one
            min_length = min(len(base_values), len(eurl_values))
            base_values = base_values[:min_length]
            eurl_values = eurl_values[:min_length]
            
            epochs = range(1, min_length + 1)
            
            axs[i].plot(epochs, base_values, label='Baseline', color='blue')
            axs[i].plot(epochs, eurl_values, label='EURL', color='orange')
            axs[i].set_title(f"{metric_name}")
            axs[i].set_ylabel(metric_name)
            
            # Add legend
            axs[i].legend()
            
            # Add grid
            axs[i].grid(True, linestyle='--', alpha=0.7)
            
            # Calculate improvement
            if metric_name.lower() in ['loss', 'perplexity']:
                # Lower is better
                improvement = (base_values[-1] - eurl_values[-1]) / base_values[-1] * 100
                better_text = "lower"
            else:
                # Higher is better
                improvement = (eurl_values[-1] - base_values[-1]) / base_values[-1] * 100
                better_text = "higher"
            
            # Add improvement text
            axs[i].text(
                0.02, 0.05,
                f"Final: EURL is {improvement:.1f}% {better_text}",
                transform=axs[i].transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
        
        # Set x-axis label for the bottom subplot
        axs[-1].set_xlabel("Epoch")
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
    
    def plot_computational_efficiency(
        self,
        sequence_lengths: List[int],
        base_times: List[float],
        eurl_times: List[float],
        title: str = "Computational Efficiency",
        filename: str = "computational_efficiency.png"
    ) -> None:
        """
        Plot computational efficiency comparison.
        
        Args:
            sequence_lengths: List of sequence lengths
            base_times: List of baseline processing times
            eurl_times: List of EURL processing times
            title: Title for the plot
            filename: Filename to save the visualization
        """
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot processing times
        plt.plot(sequence_lengths, base_times, 'o-', label='Baseline', color='blue')
        plt.plot(sequence_lengths, eurl_times, 'o-', label='EURL', color='orange')
        
        # Set title and labels
        plt.title(title)
        plt.xlabel("Sequence Length")
        plt.ylabel("Processing Time (ms)")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend()
        
        # Calculate and display speedup
        avg_speedup = sum(b/e for b, e in zip(base_times, eurl_times)) / len(base_times)
        plt.text(
            0.02, 0.05,
            f"Average Speedup: {avg_speedup:.2f}x",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8)
        )
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()


# Example usage
def main():
    visualizer = EURLVisualizer(output_dir="./eurl_visualizations")
    
    # Create sample data for demonstration
    
    # Sample attention weights
    batch_size, num_heads, seq_len = 2, 4, 10
    base_attention = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
    eurl_attention = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
    
    # Sample tokens
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    
    # Sample layer outputs
    d_model = 64
    num_layers = 4
    base_layer_outputs = [torch.randn(batch_size, seq_len, d_model) for _ in range(num_layers)]
    eurl_layer_outputs = [torch.randn(batch_size, seq_len, d_model) for _ in range(num_layers)]
    
    # Sample training metrics
    base_metrics = {
        "loss": [3.5, 3.2, 2.9, 2.7, 2.5, 2.3, 2.2, 2.1, 2.0, 1.9],
        "accuracy": [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.62, 0.64, 0.65, 0.66]
    }
    
    eurl_metrics = {
        "loss": [3.4, 3.0, 2.7, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6],
        "accuracy": [0.32, 0.43, 0.48, 0.54, 0.6, 0.65, 0.68, 0.7, 0.72, 0.74]
    }
    
    # Visualize attention patterns
    visualizer.visualize_attention_patterns(
        base_attention, tokens, layer_idx=0, head_idx=0,
        title="Baseline Attention Pattern"
    )
    
    visualizer.compare_attention_patterns(
        base_attention, eurl_attention, tokens, layer_idx=0, head_idx=0
    )
    
    # Visualize attention entropy
    visualizer.compare_attention_entropy(base_attention, eurl_attention)
    
    # Visualize inter-layer alignment
    visualizer.compare_interlayer_alignment(base_layer_outputs, eurl_layer_outputs)
    
    # Visualize training metrics
    visualizer.compare_training_metrics(
        base_metrics, eurl_metrics, ["loss", "accuracy"]
    )
    
    # Plot computational efficiency
    sequence_lengths = [32, 64, 128, 256, 512, 1024]
    base_times = [10, 22, 47, 100, 220, 480]  # Hypothetical processing times
    eurl_times = [9, 18, 36, 75, 150, 300]    # Hypothetical processing times
    
    visualizer.plot_computational_efficiency(
        sequence_lengths, base_times, eurl_times
    )
    
    print("Visualizations created in:", visualizer.output_dir)


if __name__ == "__main__":
    main()