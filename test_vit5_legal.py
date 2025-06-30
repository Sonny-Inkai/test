import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import argparse
import os

class ViT5LegalTester:
    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Initialize tester with trained model
        
        Args:
            model_path: Path to trained model checkpoint or directory
            tokenizer_path: Path to tokenizer (if different from model_path)
        """
        print(f"Loading model from: {model_path}")
        
        # Load tokenizer
        tokenizer_path = tokenizer_path or model_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        # Add special tokens (same as training)
        domain_special_tokens = [
            "<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>",
            "<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"
        ]
        
        special_tokens_dict = {'additional_special_tokens': domain_special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
    
    def extract_relations(self, text: str, max_length: int = 256, num_beams: int = 3) -> str:
        """
        Extract relations from input text
        
        Args:
            text: Input text to extract relations from
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated relations string
        """
        # Prepare input
        input_text = "extract relations: " + text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        generated_text = generated_text.replace('<pad>', '').replace('</s>', '').strip()
        
        return generated_text
    
    def parse_relations(self, relations_text: str):
        """
        Parse the generated relations text into structured format
        
        Args:
            relations_text: Generated relations string
            
        Returns:
            List of parsed relations
        """
        relations = []
        parts = relations_text.split('<')
        
        current_relation = {}
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
                    # Save previous relation
                    current_relation['tail_type'] = entity_type
                    current_relation['tail'] = entity_text.strip()
                    current_relation['relation'] = relation_type
                    relations.append(current_relation.copy())
                    current_relation = {}
                
                # Extract entity type and text
                for token in ["<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>"]:
                    if token in part:
                        entity_type = token
                        entity_text = part.replace(token, '').strip()
                        if not current_relation.get('head_type'):
                            current_relation['head_type'] = entity_type
                            current_relation['head'] = entity_text
                        break
            
            # Check for relation types
            elif any(rel in part for rel in ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]):
                for rel in ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]:
                    if rel in part:
                        relation_type = rel
                        break
        
        # Handle last relation
        if entity_type and entity_text and relation_type:
            current_relation['tail_type'] = entity_type
            current_relation['tail'] = entity_text.strip()
            current_relation['relation'] = relation_type
            relations.append(current_relation)
        
        return relations
    
    def test_samples(self, test_data_path: str = None):
        """
        Test model on sample data
        
        Args:
            test_data_path: Path to test data file
        """
        if test_data_path and os.path.exists(test_data_path):
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            print(f"Testing on {len(test_data)} samples from {test_data_path}")
            
            for i, (key, value) in enumerate(test_data.items()):
                print(f"\n{'='*80}")
                print(f"TEST SAMPLE {i+1}: {key}")
                print(f"{'='*80}")
                
                input_text = value['input_text']
                expected = value['extracted_relations']
                
                print(f"Input: {input_text[:200]}..." if len(input_text) > 200 else f"Input: {input_text}")
                print(f"\nExpected: {expected}")
                
                generated = self.extract_relations(input_text)
                print(f"Generated: {generated}")
                
                # Parse relations
                parsed = self.parse_relations(generated)
                print(f"\nParsed relations ({len(parsed)} found):")
                for j, rel in enumerate(parsed):
                    print(f"  {j+1}. {rel}")
        
        else:
            # Test with predefined samples
            test_samples = [
                "Điều 5 Nghị định 15/2020/NĐ-CP quy định về tổ chức và hoạt động của Ủy ban nhân dân xã, phường, thị trấn. Ủy ban nhân dân có trách nhiệm thực hiện các nhiệm vụ được giao.",
                
                "Luật Đầu tư 2020 có hiệu lực từ ngày 01/01/2021, áp dụng trên toàn lãnh thổ Việt Nam. Bộ Kế hoạch và Đầu tư chịu trách nhiệm hướng dẫn thực hiện Luật này.",
                
                "Thông tư 01/2022/TT-BTC của Bộ Tài chính hướng dẫn thực hiện Nghị định 123/2020/NĐ-CP về quản lý tài sản công. Thông tư có hiệu lực từ ngày 15/03/2022."
            ]
            
            print("Testing on predefined samples:")
            
            for i, text in enumerate(test_samples):
                print(f"\n{'='*80}")
                print(f"TEST SAMPLE {i+1}")
                print(f"{'='*80}")
                
                print(f"Input: {text}")
                
                generated = self.extract_relations(text)
                print(f"Generated: {generated}")
                
                # Parse relations
                parsed = self.parse_relations(generated)
                print(f"\nParsed relations ({len(parsed)} found):")
                for j, rel in enumerate(parsed):
                    print(f"  {j+1}. {rel}")

def main():
    parser = argparse.ArgumentParser(description="Test ViT5 Legal Relation Extraction Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer (if different from model)")
    parser.add_argument("--test_data", type=str, help="Path to test data JSON file")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=3, help="Number of beams for generation")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = ViT5LegalTester(args.model_path, args.tokenizer_path)
    
    # Run tests
    tester.test_samples(args.test_data)

if __name__ == "__main__":
    main() 