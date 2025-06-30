import json
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from train_vit5_legal import ViT5LegalModel, VietnameseLegalDataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
import argparse
from tqdm import tqdm

def extract_triplets_vietnamese_legal(text):
    """
    Extract triplets from generated Vietnamese legal text
    Format: <ENTITY_TYPE> Entity_Text <ENTITY_TYPE> Entity_Text <RELATION_TYPE>
    """
    triplets = []
    tokens = text.replace('<s>', '').replace('</s>', '').replace('<pad>', '').strip().split()
    
    current_triplet = {}
    current_field = None
    current_text = []
    
    entity_types = ["<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>"]
    relation_types = ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token in entity_types:
            # Save previous text if any
            if current_field and current_text:
                current_triplet[current_field] = ' '.join(current_text)
                current_text = []
            
            # Start new entity
            if 'head_type' not in current_triplet:
                current_triplet['head_type'] = token
                current_field = 'head_text'
            elif 'tail_type' not in current_triplet:
                current_triplet['tail_type'] = token
                current_field = 'tail_text'
            
        elif token in relation_types:
            # Save previous text if any
            if current_field and current_text:
                current_triplet[current_field] = ' '.join(current_text)
                current_text = []
            
            current_triplet['relation'] = token
            
            # Complete triplet
            if len(current_triplet) >= 5:  # head_type, head_text, tail_type, tail_text, relation
                triplets.append(current_triplet.copy())
            
            # Reset for next triplet
            current_triplet = {}
            current_field = None
            
        else:
            current_text.append(token)
        
        i += 1
    
    # Handle last triplet if incomplete
    if current_field and current_text:
        current_triplet[current_field] = ' '.join(current_text)
        if len(current_triplet) >= 4:  # At least head and tail
            triplets.append(current_triplet)
    
    return triplets

def evaluate_model(model_path, test_data_path, batch_size=8):
    """Evaluate the trained model on test data"""
    
    # Load model
    print(f"Loading model from {model_path}")
    model = ViT5LegalModel.load_from_checkpoint(model_path)
    model.eval()
    
    # Load test data
    test_dataset = VietnameseLegalDataset(test_data_path, model.tokenizer)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=model.tokenizer,
        model=None,
        padding=True,
        return_tensors="pt"
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    
    # Evaluate
    total_loss = 0
    num_batches = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    
    return avg_loss, perplexity

def generate_examples(model_path, examples=None, max_length=256, num_beams=3):
    """Generate examples using the trained model"""
    
    # Load model
    print(f"Loading model from {model_path}")
    model = ViT5LegalModel.load_from_checkpoint(model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    # Default examples if none provided
    if examples is None:
        examples = [
            "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp",
            "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam có trách nhiệm tổ chức thực hiện các quy định pháp luật về hòa giải ở cơ sở",
            "Bộ Tư pháp chủ trì phối hợp với các cơ quan liên quan xây dựng và ban hành văn bản hướng dẫn thực hiện pháp luật",
            "Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn của mỗi cơ quan, tổ chức",
            "Nghị định này có hiệu lực thi hành kể từ ngày 15 tháng 3 năm 2014"
        ]
    
    print("Generating examples...")
    results = []
    
    for i, text in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i+1}")
        print(f"{'='*80}")
        print(f"Input: {text}")
        print(f"{'-'*80}")
        
        # Generate
        generated = model.generate_sample(text, max_length=max_length, num_beams=num_beams)
        print(f"Generated: {generated}")
        
        # Extract triplets
        triplets = extract_triplets_vietnamese_legal(generated)
        print(f"\nExtracted Triplets:")
        for j, triplet in enumerate(triplets):
            print(f"  {j+1}. {triplet}")
        
        results.append({
            'input': text,
            'generated': generated,
            'triplets': triplets
        })
        
        print(f"{'='*80}")
    
    return results

def interactive_test(model_path):
    """Interactive testing mode"""
    
    print("Loading model for interactive testing...")
    model = ViT5LegalModel.load_from_checkpoint(model_path)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    
    print("\n" + "="*50)
    print("INTERACTIVE VIETNAMESE LEGAL RELATION EXTRACTION")
    print("="*50)
    print("Enter Vietnamese legal text (or 'quit' to exit)")
    print("Example: Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn...")
    print("-"*50)
    
    while True:
        try:
            text = input("\nInput: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.")
                continue
            
            # Generate
            print("Generating...")
            generated = model.generate_sample(text, max_length=256, num_beams=3)
            print(f"Generated: {generated}")
            
            # Extract triplets
            triplets = extract_triplets_vietnamese_legal(generated)
            print(f"\nExtracted Relations:")
            if triplets:
                for i, triplet in enumerate(triplets):
                    print(f"  {i+1}. {triplet}")
            else:
                print("  No relations extracted.")
            
            print("-"*50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test Vietnamese Legal ViT5 Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Path to test data JSON file')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'generate', 'interactive'], 
                       default='generate', help='Test mode')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Max generation length')
    parser.add_argument('--num_beams', type=int, default=3,
                       help='Number of beams for generation')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        if not args.test_data:
            print("Error: --test_data is required for evaluation mode")
            return
        evaluate_model(args.model_path, args.test_data, args.batch_size)
    
    elif args.mode == 'generate':
        generate_examples(args.model_path, max_length=args.max_length, num_beams=args.num_beams)
    
    elif args.mode == 'interactive':
        interactive_test(args.model_path)

if __name__ == "__main__":
    main() 