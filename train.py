import os
import json
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

# Define custom dataset
class VietnameseLegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length, prefix=""):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.prefix = prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = self.prefix + item["input_text"]
        target_text = item["extracted_relations"]
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        with self.tokenizer.as_target_tokenizer():
            target = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        
        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()
        labels = target["input_ids"].squeeze()
        
        # Replace padding token id with -100 for loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_text": input_text,
            "target_text": target_text
        }

# Extract triplets function for Vietnamese legal text
def extract_triplets_vietnamese(text, domain_special_tokens):
    triplets = []
    head_type, head_text, tail_type, tail_text, relation_type = '', '', '', '', ''
    text = text.strip()
    
    # Get all tokens in the text
    tokens = text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").split()
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check if token is a special token for head type
        if token in domain_special_tokens:
            head_type = token
            i += 1
            head_text = ''
            
            # Collect head text until we find another special token
            while i < len(tokens) and tokens[i] not in domain_special_tokens:
                head_text += ' ' + tokens[i]
                i += 1
            
            if i < len(tokens):
                # Found tail type token
                tail_type = tokens[i]
                i += 1
                tail_text = ''
                
                # Collect tail text until we find another special token
                while i < len(tokens) and tokens[i] not in domain_special_tokens:
                    tail_text += ' ' + tokens[i]
                    i += 1
                
                if i < len(tokens):
                    # Found relation type token
                    relation_type = tokens[i]
                    
                    # Add triplet to results
                    if head_type and head_text and tail_type and tail_text and relation_type:
                        triplets.append({
                            'head_type': head_type.strip('<>'),
                            'head': head_text.strip(),
                            'tail_type': tail_type.strip('<>'),
                            'tail': tail_text.strip(),
                            'relation': relation_type.strip('<>')
                        })
                    
                    # Reset for next triplet
                    head_type, head_text, tail_type, tail_text, relation_type = '', '', '', '', ''
        i += 1
            
    return triplets

# Lightning Module for ViT5 model
class ViT5RelationExtractionModule(pl.LightningModule):
    def __init__(self, config, tokenizer, model, domain_special_tokens):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "model"])
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.domain_special_tokens = domain_special_tokens
        
        # Define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
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
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        
        # Generate predictions for a few samples
        if batch_idx == 0:
            self._generate_and_log_predictions(batch)
        
        return loss
    
    def _generate_and_log_predictions(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # Take only the first few examples to avoid cluttering logs
        sample_size = min(3, input_ids.size(0))
        input_ids = input_ids[:sample_size]
        attention_mask = attention_mask[:sample_size]
        
        # Generate predictions
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode predictions and original inputs/targets
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        input_texts = batch["input_text"][:sample_size]
        target_texts = batch["target_text"][:sample_size]
        
        # Log examples
        for i, (input_text, target_text, generated_text) in enumerate(zip(input_texts, target_texts, generated_texts)):
            self.logger.experiment.log({
                f"example_{i}/input": input_text,
                f"example_{i}/target": target_text,
                f"example_{i}/prediction": generated_text
            })
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.config.learning_rate)
        
        # Define scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.config.warmup_steps,
            num_training_steps=self.hparams.config.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

# Callback to generate and log sample predictions during training
class GenerateSamplesCallback(pl.Callback):
    def __init__(self, every_n_steps=1000):
        super().__init__()
        self.every_n_steps = every_n_steps
    
    def on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps == 0:
            pl_module._generate_and_log_predictions(batch)

def main():
    # Define configuration
    class Config:
        model_name = "VietAI/vit5-base"
        learning_rate = 5e-5
        warmup_steps = 500
        max_steps = 10000
        max_source_length = 512
        max_target_length = 128
        train_batch_size = 8
        eval_batch_size = 8
        gradient_accumulation_steps = 4
        val_check_interval = 0.25  # Check validation every 1/4 epoch
        
    config = Config()
    
    # Define special tokens for Vietnamese legal domain
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<Effective_From>", "<Applicable_In>",
        "<Relates_To>", "<Amended_By>"
    ]
    
    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.add_tokens(domain_special_tokens, special_tokens=True)
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    data_path = "/kaggle/input/vietnamese-legal-finetune-dataset"
    finetune_file_name = "dataset.json"
    
    with open(os.path.join(data_path, finetune_file_name), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert dictionary to list
    data_list = []
    for key, value in data.items():
        data_list.append(value)
    
    # Split data into train and validation sets (90/10 split)
    np.random.seed(42)
    indices = np.random.permutation(len(data_list))
    train_size = int(0.9 * len(data_list))
    
    train_data = [data_list[i] for i in indices[:train_size]]
    val_data = [data_list[i] for i in indices[train_size:]]
    
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    
    # Create datasets
    train_dataset = VietnameseLegalDataset(
        train_data, 
        tokenizer, 
        max_source_length=config.max_source_length, 
        max_target_length=config.max_target_length
    )
    
    val_dataset = VietnameseLegalDataset(
        val_data, 
        tokenizer, 
        max_source_length=config.max_source_length, 
        max_target_length=config.max_target_length
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    pl_module = ViT5RelationExtractionModule(
        config=config,
        tokenizer=tokenizer,
        model=model,
        domain_special_tokens=domain_special_tokens
    )
    
    # Define callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="vit5-legal-re-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        ),
        LearningRateMonitor(logging_interval="step"),
        GenerateSamplesCallback(every_n_steps=1000)
    ]
    
    # Initialize logger
    logger = WandbLogger(project="vietnamese-legal-relation-extraction", name="vit5-legal-re")
    
    # Initialize trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gpus=1,  # Use 1 GPU
        precision=16,  # Use mixed precision for faster training
    )
    
    # Train model
    trainer.fit(pl_module, train_dataloader, val_dataloader)
    
    # Save the final model
    pl_module.model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")
    
    print("Training completed!")

if __name__ == "__main__":
    main()