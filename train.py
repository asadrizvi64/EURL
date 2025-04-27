import argparse
import yaml
import torch
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from eurl import EnhancedEURLTransformer, train_epoch, evaluate

def create_dataloader(dataset_name, dataset_version, split, tokenizer, max_length, batch_size):
    # Load dataset from Hugging Face
    raw_dataset = load_dataset(dataset_name, dataset_version, split=split)
    
    # Create a dataset class that tokenizes the text
    class TextDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_length):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # Get text from dataset
            text = self.dataset[idx]['text']
            if not text.strip():  # Skip empty lines
                text = "empty"
            
            # Tokenize text
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Create input and target (shifted by 1 for language modeling)
            input_ids = encodings['input_ids'].squeeze()
            target_ids = input_ids.clone()
            
            # Ensure tensors are at least 2 tokens long (for input/target shift)
            if input_ids.size(0) < 2:
                # Pad to ensure at least 2 tokens
                padded = torch.full((2,), tokenizer.pad_token_id, dtype=torch.long)
                padded[:input_ids.size(0)] = input_ids
                input_ids = padded
                target_ids = padded.clone()
            
            # Return inputs and targets (shifted by 1)
            return input_ids[:-1], target_ids[1:]
    
    # Create dataset
    dataset = TextDataset(raw_dataset, tokenizer, max_length)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        collate_fn=collate_batch,  # Custom collate function to handle variable lengths
        num_workers=2
    )
    
    return dataloader

def collate_batch(batch):
    """Custom collate function to handle variable length sequences."""
    # Separate inputs and targets
    inputs, targets = zip(*batch)
    
    # Get max length in this batch
    max_input_len = max([x.size(0) for x in inputs])
    max_target_len = max([x.size(0) for x in targets])
    
    # Create tensors of same size
    batch_size = len(inputs)
    input_tensor = torch.zeros((batch_size, max_input_len), dtype=torch.long)
    target_tensor = torch.zeros((batch_size, max_target_len), dtype=torch.long)
    
    # Fill tensors
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        input_tensor[i, :inp.size(0)] = inp
        target_tensor[i, :tgt.size(0)] = tgt
    
    return input_tensor, target_tensor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train EURL model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model based on config
    model = EnhancedEURLTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        sparse_rank=config['model']['sparse_rank'],
        kappa=config['model'].get('kappa', 0.5),
        c_const=config['model'].get('c_const', 2.0)
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Using GPT-2 tokenizer for WikiText
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    
    # Update vocab size in model if needed
    if tokenizer.vocab_size > config['model']['vocab_size']:
        print(f"Warning: Tokenizer vocab size ({tokenizer.vocab_size}) is larger than model vocab size ({config['model']['vocab_size']})")
        print(f"Reinitializing model with corrected vocab size")
        model = EnhancedEURLTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            sparse_rank=config['model']['sparse_rank'],
            kappa=config['model'].get('kappa', 0.5),
            c_const=config['model'].get('c_const', 2.0)
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Load dataset
    dataset_name = config.get('dataset', {}).get('name', 'wikitext')
    dataset_version = config.get('dataset', {}).get('version', 'wikitext-103-raw-v1')
    print(f"Loading dataset: {dataset_name}/{dataset_version}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset_name,
        dataset_version,
        'train',
        tokenizer,
        config['training']['max_seq_length'],
        config['training']['batch_size']
    )
    
    val_dataloader = create_dataloader(
        dataset_name,
        dataset_version,
        'validation',
        tokenizer,
        config['training']['max_seq_length'],
        config['training']['batch_size']
    )
    
    print(f"Loaded {len(train_dataloader)} training batches and {len(val_dataloader)} validation batches")
    
    # Save configuration and tokenizer to output directory
    with open(os.path.join(args.output, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    tokenizer.save_pretrained(args.output)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        # Train one epoch
        train_metrics = train_epoch(model, train_dataloader, optimizer, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, device)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save metrics to file
        with open(os.path.join(args.output, f'metrics_epoch_{epoch+1}.yaml'), 'w') as f:
            yaml.dump({
                'train': train_metrics,
                'validation': val_metrics
            }, f)
        
        # Save checkpoint if improved
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }, os.path.join(args.output, 'best_model.pt'))
            print(f"  Saved new best model!")
    
    # Save final model
    torch.save({
        'epoch': config['training']['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.output, 'final_model.pt'))
    
    print(f"Training complete. Model saved to {args.output}")

if __name__ == "__main__":
    main()