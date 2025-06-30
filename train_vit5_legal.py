import json
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

# Vietnamese legal domain special tokens
DOMAIN_SPECIAL_TOKENS = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
]

@dataclass
class TrainingConfig:
    # Data paths
    data_path: str = "/kaggle/input/vietnamese-legal-finetune-dataset"
    finetune_file_name: str = "dataset.json"
    
    # Model settings
    model_name: str = "VietAI/vit5-large"
    max_source_length: int = 512
    max_target_length: int = 256
    
    # Training settings
    batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0
    
    # Generation settings
    num_beams: int = 4
    early_stopping: bool = False
    no_repeat_ngram_size: int = 0
    
    # Logging
    save_top_k: int = 3
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    check_val_every_n_epoch: int = 1

class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path: str, filename: str, tokenizer, max_source_length: int, max_target_length: int, split: str = "train"):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load the JSON dataset
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to list format for easier indexing
        self.samples = []
        for key, value in self.data.items():
            self.samples.append({
                'id': key,
                'title': value['title'],
                'input_text': value['input_text'],
                'extracted_relations': value['extracted_relations']
            })
        
        # Split data (80% train, 10% val, 10% test)
        total_samples = len(self.samples)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        
        if split == "train":
            self.samples = self.samples[:train_size]
        elif split == "val":
            self.samples = self.samples[train_size:train_size + val_size]
        else:  # test
            self.samples = self.samples[train_size + val_size:]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare input text
        input_text = sample['input_text']
        target_text = sample['extracted_relations']
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Replace padding token id with -100 for loss calculation
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'target_text': target_text,
            'source_text': input_text
        }

class ViT5LegalRelationExtractor(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Load tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add domain-specific special tokens
        special_tokens_dict = {'additional_special_tokens': DOMAIN_SPECIAL_TOKENS}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
        
        # Resize token embeddings to accommodate new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # For generation tracking
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store for epoch end processing
        self.training_step_outputs.append({
            'loss': loss.detach(),
            'source_text': batch['source_text'][0] if len(batch['source_text']) > 0 else "",
            'target_text': batch['target_text'][0] if len(batch['target_text']) > 0 else ""
        })
        
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generate predictions for evaluation
        if batch_idx < 3:  # Only generate for first few batches to save time
            predictions = self.generate_relations(batch['input_ids'], batch['attention_mask'])
            
            self.validation_step_outputs.append({
                'loss': loss.detach(),
                'predictions': predictions,
                'targets': batch['target_text'],
                'sources': batch['source_text']
            })
        
        return loss

    def generate_relations(self, input_ids, attention_mask):
        """Generate relation triplets for given input"""
        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.max_target_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            predictions = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=False
            )
            
        return predictions

    def on_training_epoch_end(self):
        if len(self.training_step_outputs) > 0:
            avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
            self.log('train_epoch_loss', avg_loss)
            
            # Print sample generation every epoch
            if len(self.training_step_outputs) > 0:
                sample = self.training_step_outputs[0]
                print(f"\n=== Training Sample (Epoch {self.current_epoch}) ===")
                print(f"Source: {sample['source_text'][:200]}...")
                print(f"Target: {sample['target_text']}")
        
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
            self.log('val_epoch_loss', avg_loss)
            
            # Print sample generations
            for i, output in enumerate(self.validation_step_outputs[:2]):
                print(f"\n=== Validation Sample {i+1} (Epoch {self.current_epoch}) ===")
                print(f"Source: {output['sources'][0][:200]}...")
                print(f"Target: {output['targets'][0]}")
                print(f"Prediction: {output['predictions'][0]}")
                print("-" * 100)
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Calculate total steps for scheduler
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

def create_data_module(config: TrainingConfig):
    """Create data loaders for training, validation, and test"""
    
    # Create tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    special_tokens_dict = {'additional_special_tokens': DOMAIN_SPECIAL_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Create datasets
    train_dataset = VietnameseLegalDataset(
        config.data_path, 
        config.finetune_file_name, 
        tokenizer, 
        config.max_source_length, 
        config.max_target_length, 
        split="train"
    )
    
    val_dataset = VietnameseLegalDataset(
        config.data_path, 
        config.finetune_file_name, 
        tokenizer, 
        config.max_source_length, 
        config.max_target_length, 
        split="val"
    )
    
    test_dataset = VietnameseLegalDataset(
        config.data_path, 
        config.finetune_file_name, 
        tokenizer, 
        config.max_source_length, 
        config.max_target_length, 
        split="test"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def extract_vietnamese_legal_triplets(text: str) -> List[Dict[str, str]]:
    """
    Extract triplets from generated text using Vietnamese legal domain tokens
    Format: <ENTITY_TYPE> entity_text <ENTITY_TYPE> entity_text <RELATION_TYPE>
    """
    triplets = []
    text = text.strip()
    
    # Remove special tokens for processing
    text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
    
    tokens = text.split()
    i = 0
    
    while i < len(tokens):
        # Look for entity type tokens
        if tokens[i] in DOMAIN_SPECIAL_TOKENS and i + 1 < len(tokens):
            head_type = tokens[i]
            i += 1
            
            # Collect head entity text until next special token
            head_text = []
            while i < len(tokens) and tokens[i] not in DOMAIN_SPECIAL_TOKENS:
                head_text.append(tokens[i])
                i += 1
            
            # Look for tail entity
            if i < len(tokens) and tokens[i] in DOMAIN_SPECIAL_TOKENS:
                tail_type = tokens[i]
                i += 1
                
                # Collect tail entity text until relation token
                tail_text = []
                while i < len(tokens) and tokens[i] not in DOMAIN_SPECIAL_TOKENS:
                    tail_text.append(tokens[i])
                    i += 1
                
                # Look for relation
                if i < len(tokens) and tokens[i] in DOMAIN_SPECIAL_TOKENS:
                    relation = tokens[i]
                    
                    if head_text and tail_text:
                        triplets.append({
                            'head': ' '.join(head_text).strip(),
                            'head_type': head_type,
                            'tail': ' '.join(tail_text).strip(),
                            'tail_type': tail_type,
                            'relation': relation
                        })
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1
    
    return triplets

def evaluate_model(model: ViT5LegalRelationExtractor, test_loader: DataLoader):
    """Evaluate the model on test set"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)
            
            # Calculate loss
            outputs = model.forward(input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()
            
            # Generate predictions
            predictions = model.generate_relations(input_ids, attention_mask)
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['target_text'])
    
    avg_loss = total_loss / len(test_loader)
    
    # Print some examples
    print(f"\n=== Test Results ===")
    print(f"Average Loss: {avg_loss:.4f}")
    
    for i in range(min(5, len(all_predictions))):
        print(f"\nExample {i+1}:")
        print(f"Target: {all_targets[i]}")
        print(f"Prediction: {all_predictions[i]}")
        
        # Extract and print triplets
        pred_triplets = extract_vietnamese_legal_triplets(all_predictions[i])
        target_triplets = extract_vietnamese_legal_triplets(all_targets[i])
        
        print(f"Predicted Triplets: {pred_triplets}")
        print(f"Target Triplets: {target_triplets}")
        print("-" * 80)

def main():
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Configuration
    config = TrainingConfig()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_module(config)
    
    # Initialize model
    model = ViT5LegalRelationExtractor(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor,
        dirpath='./checkpoints',
        filename='vit5-legal-{epoch:02d}-{val_loss:.2f}',
        save_top_k=config.save_top_k,
        mode=config.monitor_mode,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,
        patience=3,
        verbose=True,
        mode=config.monitor_mode
    )
    
    # Set up logger
    logger = TensorBoardLogger(save_dir='./logs', name='vit5_legal_extraction')
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        gradient_clip_val=config.gradient_clip_val,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    print("Starting testing...")
    trainer.test(model, test_loader)
    
    # Additional evaluation
    evaluate_model(model, test_loader)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main() 