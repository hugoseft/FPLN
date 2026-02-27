from Segmentacion import *

if __name__ == "__main__":

    with open("training_sentences.txt", "r", encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]

    with open("test_sentences.txt", "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    corpus_train_completo = " ".join(train_lines)

    tamanos_vocabulario = [100, 150, 200]

    for size in tamanos_vocabulario:
        print(f"\n=========================================")
        print(f"   EVALUANDO TAMAÑO DE VOCABULARIO: {size}")
        print(f"=========================================")
        
        # WordPiece
        print(f"\n--- WordPiece (Vocab: {size}) ---")
        wp_tokenizer = TokenizadorWordPiece(vocab_size=size)
        wp_tokenizer.train(corpus_train_completo)
        print(f"Vocabulario WordPiece ({size}):\n{wp_tokenizer.vocab}\n")
        
        for line in test_lines:
            print(f"Input: '{line}' -> Tokens: {wp_tokenizer.tokenize(line)}")

        # BPE
        print(f"\n--- BPE (Vocab: {size}) ---")
        bpe_tokenizer = TokenizadorBPE(vocab_size=size)
        bpe_tokenizer.train(corpus_train_completo)
        print(f"Vocabulario BPE ({size}):\n{bpe_tokenizer.vocab}\n")
        
        for line in test_lines:
            print(f"Input: '{line}' -> Tokens: {bpe_tokenizer.tokenize(line)}")

