import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_lightning.callbacks import Callback

# Vietnamese Legal Domain Special Tokens
DOMAIN_SPECIAL_TOKENS = [
    "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
    "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
]

class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length=512, max_target_length=256):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
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
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['extracted_relations'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace pad tokens with -100)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class ViT5LegalModel(pl.LightningModule):
    def __init__(self, model_name="VietAI/vit5-base", learning_rate=5e-5, weight_decay=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens_dict = {'additional_special_tokens': DOMAIN_SPECIAL_TOKENS}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens")
        
        # Load model config and model
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
        
        # Resize token embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
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
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def generate_sample(self, text, max_length=256, num_beams=3):
        """Generate relation extraction for sample text"""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            generated_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            return generated_text

class GenerateExamplesCallback(Callback):
    def __init__(self, test_texts, every_n_epochs=1):
        self.test_texts = test_texts
        self.every_n_epochs = every_n_epochs
    
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            print(f"\n{'='*50}")
            print(f"EPOCH {trainer.current_epoch + 1} - GENERATION EXAMPLES")
            print(f"{'='*50}")
            
            for i, text in enumerate(self.test_texts):
                print(f"\nExample {i+1}:")
                print(f"Input: {text[:100]}...")
                generated = pl_module.generate_sample(text)
                print(f"Generated: {generated}")
                print("-" * 50)

def create_data_module(data_path, tokenizer, batch_size=8, train_split=0.8):
    """Create train/val dataloaders"""
    
    # Load full dataset
    dataset = VietnameseLegalDataset(data_path, tokenizer)
    
    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None,  # We'll handle padding manually
        padding=True,
        return_tensors="pt"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=data_collator
    )
    
    return train_loader, val_loader

def main():
    # Configuration
    config = {
        'model_name': "VietAI/vit5-base",
        'data_path': "/kaggle/input/vietnamese-legal-finetune-dataset/dataset.json",
        'batch_size': 8,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_epochs': 10,
        'accumulate_grad_batches': 4,
        'precision': 16,  # Mixed precision for faster training
        'gradient_clip_val': 1.0,
        'train_split': 0.8,
        'save_top_k': 3,
        'monitor': 'val_loss',
        'mode': 'min'
    }
    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Initialize model
    print("Initializing ViT5 Legal Model...")
    model = ViT5LegalModel(
        model_name=config['model_name'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_module(
        data_path=config['data_path'],
        tokenizer=model.tokenizer,
        batch_size=config['batch_size'],
        train_split=config['train_split']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Sample texts for generation testing
    test_texts = [
        "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở",
        "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam có trách nhiệm tổ chức thực hiện các quy định pháp luật"
    ]
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='./checkpoints',
            filename='vit5-legal-{epoch:02d}-{val_loss:.2f}',
            monitor=config['monitor'],
            mode=config['mode'],
            save_top_k=config['save_top_k'],
            verbose=True,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        GenerateExamplesCallback(test_texts, every_n_epochs=1)
    ]
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,  # Use devices instead of gpus
        precision=config['precision'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        gradient_clip_val=config['gradient_clip_val'],
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=False,  # Disable wandb as requested
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("Training completed!")
    print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    
    # Test final model
    print("\nTesting final model:")
    for text in test_texts:
        result = model.generate_sample(text)
        print(f"Input: {text}")
        print(f"Output: {result}")
        print("-" * 50)

if __name__ == "__main__":
    main() 