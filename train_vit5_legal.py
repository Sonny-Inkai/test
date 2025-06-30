import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoConfig,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Custom dataset class for Vietnamese legal data
class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_source_length: int = 512, max_target_length: int = 256):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to list format
        self.samples = []
        for key, value in self.data.items():
            self.samples.append({
                'input_text': value['input_text'],
                'extracted_relations': value['extracted_relations']
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            sample['input_text'],
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        targets = self.tokenizer(
            sample['extracted_relations'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs.input_ids.flatten(),
            'attention_mask': inputs.attention_mask.flatten(),
            'labels': targets.input_ids.flatten()
        }

# Custom data module
class VietnameseLegalDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, finetune_file_name: str, tokenizer, 
                 batch_size: int = 8, max_source_length: int = 512, max_target_length: int = 256):
        super().__init__()
        self.data_path = data_path
        self.finetune_file_name = finetune_file_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def setup(self, stage=None):
        full_path = os.path.join(self.data_path, self.finetune_file_name)
        
        # Load full dataset
        dataset = VietnameseLegalDataset(
            full_path, self.tokenizer, 
            self.max_source_length, self.max_target_length
        )
        
        # Split data (80% train, 20% val)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=2
        )

# Lightning module for ViT5
class ViT5LegalExtractionModule(pl.LightningModule):
    def __init__(self, model_name: str = "VietAI/vit5-base", learning_rate: float = 5e-5, 
                 warmup_steps: int = 1000, total_steps: int = 10000):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for Vietnamese legal domain
        domain_special_tokens = [
            "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
            "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
        ]
        
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': domain_special_tokens
        })
        
        # Load model
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        
        # Resize embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        # Generate predictions for evaluation
        if batch_idx == 0:  # Only for first batch to save time
            generated = self.model.generate(
                input_ids=batch['input_ids'][:2],  # Take first 2 samples
                attention_mask=batch['attention_mask'][:2],
                max_length=256,
                num_beams=3,
                early_stopping=True
            )
            
            # Decode predictions and targets
            pred_texts = self.tokenizer.batch_decode(generated, skip_special_tokens=False)
            target_texts = self.tokenizer.batch_decode(batch['labels'][:2], skip_special_tokens=False)
            
            # Log samples
            for i, (pred, target) in enumerate(zip(pred_texts, target_texts)):
                self.logger.experiment.add_text(f'pred_{i}', pred, self.current_epoch)
                self.logger.experiment.add_text(f'target_{i}', target, self.current_epoch)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def generate_sample(self, text: str) -> str:
        """Generate relation extraction for a given text"""
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=3,
                early_stopping=True,
                do_sample=False
            )
        
        result = self.tokenizer.decode(generated[0], skip_special_tokens=False)
        return result

# Callback for generating samples
class GenerateSamplesCallback(pl.Callback):
    def __init__(self, sample_texts: List[str], every_n_epochs: int = 1):
        super().__init__()
        self.sample_texts = sample_texts
        self.every_n_epochs = every_n_epochs
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            print(f"\n=== Generation Examples (Epoch {trainer.current_epoch}) ===")
            pl_module.eval()
            
            for i, text in enumerate(self.sample_texts):
                print(f"\nSample {i+1}:")
                print(f"Input: {text[:200]}...")
                try:
                    generated = pl_module.generate_sample(text)
                    print(f"Generated: {generated}")
                except Exception as e:
                    print(f"Generation failed: {e}")
            
            pl_module.train()
            print("=" * 50)

def main():
    # Configuration
    data_path = "/kaggle/input/vietnamese-legal-finetune-dataset"
    finetune_file_name = "dataset.json"
    model_name = "VietAI/vit5-base"
    
    # Training parameters
    batch_size = 8
    learning_rate = 5e-5
    max_epochs = 10
    accumulate_grad_batches = 4
    
    # Sample texts for generation callback
    sample_texts = [
        "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở",
        "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam có nhiệm vụ hướng dẫn và tổ chức thực hiện"
    ]
    
    # Initialize model
    model = ViT5LegalExtractionModule(
        model_name=model_name,
        learning_rate=learning_rate,
        warmup_steps=1000,
        total_steps=max_epochs * 100  # Rough estimate
    )
    
    # Initialize data module
    data_module = VietnameseLegalDataModule(
        data_path=data_path,
        finetune_file_name=finetune_file_name,
        tokenizer=model.tokenizer,
        batch_size=batch_size
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='vit5-legal-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step'),
        GenerateSamplesCallback(sample_texts, every_n_epochs=1)
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=2,  # Use 2 GPUs (T4x2)
        accelerator='gpu',
        strategy='ddp',
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate twice per epoch
        precision=16,  # Mixed precision training
        enable_progress_bar=True
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Save final model
    model.model.save_pretrained("vit5_legal_final")
    model.tokenizer.save_pretrained("vit5_legal_final")
    
    print("Training completed! Model saved to 'vit5_legal_final'")

if __name__ == "__main__":
    main() 