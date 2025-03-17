from llama_cpp import Llama

print("Testing llama.cpp using llama-cpp-python package with NSFW-3B model")

try:
    # Load model in vocab-only mode first
    print("\nAttempting to load model vocabulary only...")
    model_path = "models/nsfw-3b-q4_k_m.gguf"
    
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        vocab_only=True,
        verbose=True
    )
    
    print("\nModel loaded successfully!")
    print(f"Model vocabulary size: {llm.n_vocab()}")
    print(f"Context size: {llm.n_ctx()}")
    
    # Test tokenization
    test_text = "Hello, world!"
    tokens = llm.tokenize(text=test_text.encode())
    print(f"\nTokenization test:")
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # Test detokenization
    decoded = llm.detokenize(tokens).decode()
    print(f"Decoded text: {decoded}")
    
    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error: {e}") 