from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Keeps your laptop responsive by limiting CPU usage
torch.set_num_threads(4) 

def run_chat():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading model... (this takes about 1 minute)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

    print("\n--- CHAT MODE ACTIVE (Type 'exit' to stop) ---")

    while True:
        user_prompt = input("\nYour Question: ")
        
        if user_prompt.lower() in ["exit", "quit"]:
            break

        print("Generating...")
        inputs = tokenizer(user_prompt, return_tensors="pt").to("cpu")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response to end on a full sentence
        if "." in response:
            response = response[:response.rfind(".")+1]
            
        print(f"\n[Response]:\n{response}")

if __name__ == "__main__":
    run_chat()
