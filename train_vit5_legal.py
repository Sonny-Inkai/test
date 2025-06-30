import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import os
from typing import Dict, List, Any
import random

class VietnameseLegalDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: T5Tokenizer, max_source_length: int = 512, max_target_length: int = 256):
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
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Prepare input text (prefix for T5)
        input_text = "extract relations: " + sample['input_text']
        target_text = sample['extracted_relations']
        
        # Tokenize
        source = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source['input_ids'].squeeze(),
            'attention_mask': source['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze(),
            'target_attention_mask': target['attention_mask'].squeeze()
        }

class ViT5LegalExtractionModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "VietAI/vit5-base",
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,  # Add label smoothing like REBEL
        ignore_pad_token_for_loss: bool = True,  # Add pad token ignoring
        domain_special_tokens: List[str] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # FIX T5 tokenizer issues
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to: {self.tokenizer.pad_token}")
        
        # Add special tokens
        if domain_special_tokens is None:
            domain_special_tokens = [
                "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
                "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
            ]
        
        # Add special tokens to tokenizer
        special_tokens_dict = {'additional_special_tokens': domain_special_tokens}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_toks} special tokens")
        print(f"Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token_id: {self.tokenizer.eos_token_id}")
        
        # Load model and resize embeddings
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Training params
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        
        # Set up loss function - simpler approach for T5
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        # For tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
    
    def label_smoothed_nll_loss(self, lprobs, target, epsilon, ignore_index=-100):
        """Label smoothed loss function adapted from REBEL"""
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def forward(self, input_ids, attention_mask, labels=None):
        """Simple T5 forward pass - let T5 handle everything first"""
        if labels is None:
            # Generation mode
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Training mode - use T5's built-in loss computation
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def training_step(self, batch, batch_idx):
        # Prepare labels like REBEL - but T5 doesn't need shift_tokens_left
        labels = batch['labels'].clone()
        
        # For T5, we don't need to create decoder_input_ids explicitly
        # T5 handles this internally
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        
        loss = outputs['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.training_step_outputs.append(loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Same proper loss computation for validation
        labels = batch['labels'].clone()
        
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=labels
        )
        
        loss = outputs['loss']
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.validation_step_outputs.append(loss.detach())
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        # Only print on rank 0 to avoid duplicate output
        if self.trainer.is_global_zero:
            print(f"\nTrain Epoch {self.current_epoch} - Average Loss: {avg_loss:.4f}")
        self.training_step_outputs.clear()
        
        # Generate sample after each epoch - only on rank 0
        if self.trainer.is_global_zero:
            self.generate_sample()
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        if self.trainer.is_global_zero:
            print(f"Val Epoch {self.current_epoch} - Average Loss: {avg_loss:.4f}")
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def parse_relations_output(self, relations_text: str):
        """Parse generated relations into structured format"""
        relations = []
        parts = relations_text.split('<')
        
        current_triplet = {}
        entity_type = None
        entity_text = ""
        relation_type = None
        
        for part in parts:
            if not part:
                continue
                
            part = '<' + part
            
            # Check for entity types
            if any(token in part for token in ["<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>"]):
                if entity_type and entity_text and relation_type:
                    # Complete previous triplet
                    current_triplet['tail_type'] = entity_type.replace('<', '').replace('>', '')
                    current_triplet['tail'] = entity_text.strip()
                    current_triplet['relation'] = relation_type.replace('<', '').replace('>', '')
                    relations.append(current_triplet.copy())
                    current_triplet = {}
                
                # Extract entity type and text
                for token in ["<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>"]:
                    if token in part:
                        entity_type = token
                        entity_text = part.replace(token, '').strip()
                        if not current_triplet.get('head_type'):
                            current_triplet['head_type'] = entity_type.replace('<', '').replace('>', '')
                            current_triplet['head'] = entity_text
                        break
            
            # Check for relation types
            elif any(rel in part for rel in ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]):
                for rel in ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]:
                    if rel in part:
                        relation_type = rel
                        break
        
        # Handle last triplet
        if entity_type and entity_text and relation_type:
            current_triplet['tail_type'] = entity_type.replace('<', '').replace('>', '')
            current_triplet['tail'] = entity_text.strip()
            current_triplet['relation'] = relation_type.replace('<', '').replace('>', '')
            relations.append(current_triplet)
        
        return relations

    def generate_sample(self):
        """Generate sample to check model progress with multiple test cases"""
        self.model.eval()
        
        # Multiple test samples from the dataset
        test_samples = [
            {
                "input": "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở.",
                "expected": "<LEGAL_PROVISION> Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN <ORGANIZATION> Mặt trận Tổ quốc Việt Nam <Relates_To>"
            },
            {
                "input": "Nghị định 52/2021/NĐ-CP của Chính phủ quy định chi tiết thi hành một số điều của Luật An toàn thông tin mạng. Nghị định này có hiệu lực từ ngày 01/10/2021 và áp dụng tại toàn bộ lãnh thổ Việt Nam.",
                "expected": "<LEGAL_PROVISION> Nghị định 52/2021/NĐ-CP <LEGAL_PROVISION> Luật An toàn thông tin mạng <Relates_To> <LEGAL_PROVISION> Nghị định 52/2021/NĐ-CP <DATE/TIME> 01/10/2021 <Effective_From> <LEGAL_PROVISION> Nghị định 52/2021/NĐ-CP <LOCATION> Việt Nam <Applicable_In> <ORGANIZATION> Bộ Thông tin và Truyền thông <LEGAL_PROVISION> Nghị định 52/2021/NĐ-CP <Relates_To>"
            }
        ]
        
        print(f"\n{'🔥'*50}")
        print(f"🧠 EPOCH {self.current_epoch} - MODEL GENERATION TESTING 🧠")
        print(f"{'🔥'*50}")
        
        for i, sample in enumerate(test_samples):
            print(f"\n{'='*80}")
            print(f"📋 TEST CASE {i+1}")
            print(f"{'='*80}")
            
            input_text = "extract relations: " + sample["input"]
            expected = sample["expected"]
            
            # Truncate long input for display
            display_input = sample["input"][:150] + "..." if len(sample["input"]) > 150 else sample["input"]
            print(f"📝 INPUT: {display_input}")
            
            # Generate
            inputs = self.tokenizer(
                input_text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                    min_length=10,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    no_repeat_ngram_size=2,  # Prevent repetition
                    repetition_penalty=1.2,   # Penalize repetition
                    length_penalty=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            generated_text = generated_text.replace('<pad>', '').replace('</s>', '').strip()
            
            print(f"\n🎯 EXPECTED: {expected}")
            print(f"🤖 GENERATED: {generated_text}")
            
            # Parse and display structured relations
            generated_relations = self.parse_relations_output(generated_text)
            expected_relations = self.parse_relations_output(expected)
            
            print(f"\n📊 PARSED RELATIONS:")
            print(f"Expected ({len(expected_relations)} relations):")
            for j, rel in enumerate(expected_relations):
                print(f"  ✅ {j+1}. [{rel.get('head_type', '?')}] {rel.get('head', '?')} --[{rel.get('relation', '?')}]--> [{rel.get('tail_type', '?')}] {rel.get('tail', '?')}")
            
            print(f"\nGenerated ({len(generated_relations)} relations):")
            for j, rel in enumerate(generated_relations):
                print(f"  🤖 {j+1}. [{rel.get('head_type', '?')}] {rel.get('head', '?')} --[{rel.get('relation', '?')}]--> [{rel.get('tail_type', '?')}] {rel.get('tail', '?')}")
            
            # Simple accuracy check
            if generated_text.strip() == expected.strip():
                print(f"\n🎉 PERFECT MATCH!")
            elif len(generated_relations) == len(expected_relations):
                print(f"\n✅ SAME NUMBER OF RELATIONS")
            elif len(generated_relations) > 0:
                print(f"\n⚠️ PARTIAL GENERATION")
            else:
                print(f"\n❌ NO RELATIONS GENERATED")
            
            print(f"{'─'*80}")
        
        print(f"\n{'🎯'*50}")
        print(f"🧠 END OF EPOCH {self.current_epoch} TESTING 🧠")
        print(f"{'🎯'*50}\n")
        
        self.model.train()

def create_data_splits(data_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train/val/test"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = list(data.items())
    random.shuffle(items)
    
    n = len(items)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = dict(items[:train_size])
    val_data = dict(items[train_size:train_size + val_size])
    test_data = dict(items[train_size + val_size:])
    
    # Save splits
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open('val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    with open('test_data.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Data splits created: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return 'train_data.json', 'val_data.json', 'test_data.json'

def main():
    # Configuration
    config = {
        'data_path': "/kaggle/input/vietnamese-legal-finetune-dataset",
        'finetune_file_name': "dataset.json",
        'model_name': "VietAI/vit5-base",
        'batch_size': 8,
        'learning_rate': 5e-5,  # Lower LR - T5 needs smaller LR than BART
        'max_epochs': 10,
        'max_steps': 10000,
        'warmup_steps': 1000,
        'max_source_length': 512,
        'max_target_length': 256,
        'weight_decay': 0.01,
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 4,
        'precision': 16,  # Mixed precision
        'num_workers': 2,
        # Start simple - add complexity later
        'label_smoothing': 0.0,  # Disable label smoothing first
        'ignore_pad_token_for_loss': True
    }
    
    # Special tokens for Vietnamese legal domain
    domain_special_tokens = [
        "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
        "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
    ]
    
    print("Starting Vietnamese Legal Joint Entity and Relation Extraction Training")
    print(f"Using model: {config['model_name']}")
    print(f"Special tokens: {domain_special_tokens}")
    
    # Data preparation
    data_file_path = os.path.join(config['data_path'], config['finetune_file_name'])
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return
    
    # Create data splits
    train_file, val_file, test_file = create_data_splits(data_file_path)
    
    # Initialize model
    model = ViT5LegalExtractionModel(
        model_name=config['model_name'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        max_steps=config['max_steps'],
        weight_decay=config['weight_decay'],
        label_smoothing=config.get('label_smoothing', 0.1),  # Add label smoothing
        ignore_pad_token_for_loss=config.get('ignore_pad_token_for_loss', True),
        domain_special_tokens=domain_special_tokens
    )
    
    # Create datasets
    train_dataset = VietnameseLegalDataset(
        train_file, 
        model.tokenizer, 
        config['max_source_length'], 
        config['max_target_length']
    )
    
    val_dataset = VietnameseLegalDataset(
        val_file, 
        model.tokenizer, 
        config['max_source_length'], 
        config['max_target_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='vit5-legal-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        verbose=True
    )
    
    # Trainer with fixed GPU configuration for newer PyTorch Lightning
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=2 if torch.cuda.is_available() else 1,  # T4x2 GPUs
        max_epochs=config['max_epochs'],
        max_steps=config['max_steps'],
        precision=config['precision'],
        gradient_clip_val=config['gradient_clip_val'],
        accumulate_grad_batches=config['accumulate_grad_batches'],
        callbacks=[checkpoint_callback, early_stopping],
        val_check_interval=1.0,  # Check validation only at end of each epoch - FASTER!
        log_every_n_steps=100,   # Reduce logging frequency
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=2  # Only 2 sanity validation steps instead of default
    )
    
    print(f"\nTraining configuration:")
    print(f"- Accelerator: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"- Devices: {trainer.num_devices}")
    print(f"- Max epochs: {config['max_epochs']}")
    print(f"- Batch size: {config['batch_size']}")
    print(f"- Learning rate: {config['learning_rate']}")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    
    # Start training
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    print("\nTraining completed!")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Test the best model
    print("\nTesting final model...")
    best_model = ViT5LegalExtractionModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        domain_special_tokens=domain_special_tokens,
        label_smoothing=config.get('label_smoothing', 0.1),
        ignore_pad_token_for_loss=config.get('ignore_pad_token_for_loss', True)
    )
    best_model.generate_sample()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # Run training
    main() 