import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

def load_model(model_path: str):
    """Load the trained ViT5 model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def extract_relations(text: str, model, tokenizer, max_length: int = 256):
    """Extract relations from Vietnamese legal text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Generate predictions
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=3,
            early_stopping=True,
            do_sample=False,
            temperature=1.0
        )
    
    # Decode result
    result = tokenizer.decode(generated[0], skip_special_tokens=False)
    return result

def parse_extracted_relations(extracted_text: str):
    """Parse the extracted relations text into structured format"""
    relations = []
    
    # Simple parsing logic - you might need to adjust based on your exact format
    tokens = extracted_text.replace('<pad>', '').replace('</s>', '').split()
    
    current_entity_type = None
    current_entity_text = ""
    current_relation = None
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token in ["<ORGANIZATION>", "<LOCATION>", "<DATE/TIME>", "<LEGAL_PROVISION>"]:
            if current_entity_type and current_entity_text and current_relation:
                # Store the previous relation
                relations.append({
                    'head_type': current_entity_type,
                    'head_text': current_entity_text.strip(),
                    'relation': current_relation,
                    'tail_type': token,
                    'tail_text': ""
                })
            current_entity_type = token
            current_entity_text = ""
            
        elif token in ["<Effective_From>", "<Applicable_In>", "<Relates_To>", "<Amended_By>"]:
            current_relation = token
            
        else:
            if current_entity_type:
                current_entity_text += " " + token
        
        i += 1
    
    return relations

def main():
    # Test examples
    test_examples = [
        "Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở Nguyên tắc phối hợp 1. Việc phối hợp hoạt động được thực hiện trên cơ sở chức năng, nhiệm vụ, quyền hạn, bảo đảm vai trò, trách nhiệm của mỗi cơ quan, tổ chức. 2. Phát huy vai trò nòng cốt của Mặt trận Tổ quốc Việt Nam và các tổ chức thành viên của Mặt trận; tăng cường tính chủ động, tích cực của mỗi cơ quan, tổ chức trong công tác hòa giải ở cơ sở.",
        
        "Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam có nhiệm vụ hướng dẫn và tổ chức thực hiện các quy định của pháp luật về hòa giải ở cơ sở theo đề nghị của Bộ Tư pháp.",
        
        "Luật này có hiệu lực từ ngày 01 tháng 01 năm 2015 và áp dụng trên toàn lãnh thổ Việt Nam."
    ]
    
    # Load the trained model
    model_path = "vit5_legal_final"  # Path to your saved model
    
    try:
        print("Loading model...")
        model, tokenizer = load_model(model_path)
        print("Model loaded successfully!")
        
        # Test the model
        for i, text in enumerate(test_examples):
            print(f"\n{'='*80}")
            print(f"Test Example {i+1}:")
            print(f"Input: {text[:150]}...")
            
            # Extract relations
            extracted = extract_relations(text, model, tokenizer)
            print(f"\nRaw Output: {extracted}")
            
            # Parse relations
            relations = parse_extracted_relations(extracted)
            if relations:
                print(f"\nParsed Relations:")
                for j, rel in enumerate(relations):
                    print(f"  {j+1}. {rel['head_type']} '{rel['head_text']}' {rel['relation']} {rel['tail_type']} '{rel['tail_text']}'")
            else:
                print("\nNo relations extracted or parsing failed.")
                
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model is trained and saved properly.")

if __name__ == "__main__":
    main() 