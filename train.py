import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq
)
from typing import Dict, List, Any
import wandb

# Domain-specific special tokens for Vietnamese legal documents
DOMAIN_SPECIAL_TOKENS = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
]

class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path: str, filename: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        file_path = os.path.join(data_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to list format
        self.examples = []
        for key, value in self.data.items():
            self.examples.append({
                'input_text': value['input_text'],
                'extracted_relations': value['extracted_relations']
            })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['extracted_relations'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class VietnameseLegalT5(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        max_epochs: int = 10,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize tokenizer and add special tokens
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': DOMAIN_SPECIAL_TOKENS})
        
        # Initialize model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return {'val_loss': loss}
    
    def generate_sample(self, input_text: str, max_length: int = 512):
        """Generate relation extraction for a sample text"""
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer(
                input_text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            ).input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            return decoded
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        
        # Calculate total steps for scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = min(self.warmup_steps, total_steps // 10)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
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

class VietnameseLegalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        filename: str,
        tokenizer,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        max_length: int = 512,
        train_val_split: float = 0.9
    ):
        super().__init__()
        self.data_path = data_path
        self.filename = filename
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_length = max_length
        self.train_val_split = train_val_split
        
    def setup(self, stage=None):
        # Load full dataset
        full_dataset = VietnameseLegalDataset(
            self.data_path, 
            self.filename, 
            self.tokenizer, 
            self.max_length
        )
        
        # Split into train and validation
        train_size = int(len(full_dataset) * self.train_val_split)
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

class GenerationCallback(pl.Callback):
    def __init__(self, sample_texts: List[str], every_n_epochs: int = 1):
        self.sample_texts = sample_texts
        self.every_n_epochs = every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            print(f"\n=== Generation Samples at Epoch {trainer.current_epoch} ===")
            
            for i, text in enumerate(self.sample_texts):
                generated = pl_module.generate_sample(text)
                print(f"\nSample {i+1}:")
                print(f"Input: {text[:100]}...")
                print(f"Generated: {generated}")
                
                # Log to wandb if available
                if hasattr(trainer.logger, 'experiment'):
                    trainer.logger.experiment.log({
                        f"sample_{i+1}_input": text[:100],
                        f"sample_{i+1}_output": generated,
                        "epoch": trainer.current_epoch
                    })

def main():
    # Configuration
    CONFIG = {
        'model_name': "VietAI/vit5-base",
        'data_path': "/kaggle/input/vietnamese-legal-finetune-dataset",
        'filename': "dataset.json",
        'learning_rate': 3e-4,
        'train_batch_size': 4,  # Adjusted for T4x2
        'eval_batch_size': 4,
        'max_epochs': 10,
        'max_length': 512,
        'warmup_steps': 500,
        'accumulate_grad_batches': 4,  # Effective batch size = 4 * 4 = 16
        'precision': 16,  # Mixed precision for faster training
        'gradient_clip_val': 1.0
    }
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Initialize model
    model = VietnameseLegalT5(
        model_name=CONFIG['model_name'],
        learning_rate=CONFIG['learning_rate'],
        warmup_steps=CONFIG['warmup_steps'],
        train_batch_size=CONFIG['train_batch_size'],
        eval_batch_size=CONFIG['eval_batch_size']
    )
    
    # Initialize data module
    data_module = VietnameseLegalDataModule(
        data_path=CONFIG['data_path'],
        filename=CONFIG['filename'],
        tokenizer=model.tokenizer,
        train_batch_size=CONFIG['train_batch_size'],
        eval_batch_size=CONFIG['eval_batch_size'],
        max_length=CONFIG['max_length']
    )
    
    # Sample texts for generation testing
    sample_texts = [
        "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở",
        "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam thực hiện nhiệm vụ theo quy định của pháp luật"
    ]
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='vietnamese-legal-t5-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step'),
        GenerationCallback(sample_texts, every_n_epochs=1)
    ]
    
    # Logger
    logger = WandbLogger(
        project="vietnamese-legal-relation-extraction",
        name="vit5-legal-rebel"
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        accelerator='gpu',
        devices=2,  # T4x2
        strategy='ddp',  # Distributed training
        precision=CONFIG['precision'],
        accumulate_grad_batches=CONFIG['accumulate_grad_batches'],
        gradient_clip_val=CONFIG['gradient_clip_val'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.5,  # Validate twice per epoch
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test generation after training
    print("\n=== Final Test Generation ===")
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        best_model = VietnameseLegalT5.load_from_checkpoint(best_model_path)
        for text in sample_texts:
            result = best_model.generate_sample(text)
            print(f"Input: {text}")
            print(f"Generated: {result}")
            print("-" * 50)

if __name__ == "__main__":
    main() 