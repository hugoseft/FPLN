#TODO
import regex as re 
import pickle
import string
from ModeloClasificacionSupervisada import extraer_caracteristicas

def Token_espacios(texto):
    tokens = texto.split(" ")
    return tokens


def Token_puntuacion(texto):
    pattern = r"\w+|[^\w\s]|\p{So}"
    tokens = re.findall(pattern, texto, flags=re.UNICODE)
    return tokens


def Token_n_gramas(texto, n, tokenizacion_func=Token_espacios):
    tokens = tokenizacion_func(texto)
    n_gramas = []
    for i in range(len(tokens) - n + 1):
        n_grama = tokens[i:i+n]
        n_grama = " ".join(n_grama)
        n_gramas.append(n_grama)
    return n_gramas


def Token_clas_superv(texto):
    tokens = []
    
    try:
        with open("modelo_token.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("vectorizador.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return ["Error: Ejecuta primero entrenar.py para generar el modelo"]

    texto_sin_espacios = texto.replace(" ","")
    features_list = []
    
    for i, char in enumerate(texto_sin_espacios):
        next_c = texto_sin_espacios[i+1] if i+1 < len(texto_sin_espacios) else "EOF"
        features_list.append(extraer_caracteristicas(char, next_c))
    
    
    X_test = vectorizer.transform(features_list) 
    predicciones = clf.predict(X_test)
    
    
    palabra_actual = ""
    for i, char in enumerate(texto_sin_espacios):
        palabra_actual += char
        
        if predicciones[i] == 1:
            tokens.append(palabra_actual)
            palabra_actual = ""
            
    if palabra_actual:
        tokens.append(palabra_actual)
        
    return tokens

class TokenizadorWordPiece:
    def __init__(self, vocab_size=150):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.splits = {}

    def train(self, corpus_text):
       
        words_freq = {}
        for word in corpus_text.split():
            words_freq[word] = words_freq.get(word, 0) + 1

        self.vocab.add("[UNK]") 
        
       
        for word in words_freq.keys():
            split = []
            for i, char in enumerate(word):
                token = char if i == 0 else "##" + char 
                split.append(token)
                self.vocab.add(token)
            self.splits[word] = split 


        while len(self.vocab) < self.vocab_size:
            pairs = {}
            for word, freq in words_freq.items():
                split = self.splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get) 
            p1, p2 = best_pair
            
           
            new_token = p1 + p2.replace("##", "")
            self.vocab.add(new_token) 

            
            for word in words_freq.keys():
                split = self.splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == p1 and split[i+1] == p2:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                self.splits[word] = new_split

    def tokenize(self, text):
        tokens = []
        for word in text.split(): 
            if word in self.vocab:
                tokens.append(word)
                continue
            
            sub_tokens = []
            start = 0
            is_unknown = False
            
            while start < len(word):
                end = len(word)
                match = None
                min_len = start + 2 if start == 0 else start + 1
                
                while end >= min_len:
                    sub = word[start:end]
                    if start > 0:
                        sub = "##" + sub
                    if sub in self.vocab:
                        match = sub
                        break
                    end -= 1
                
                if match is None:
                    is_unknown = True
                    break
                sub_tokens.append(match)
                start = end
                
            if is_unknown:
                tokens.append("[UNK]") 
            else:
                tokens.extend(sub_tokens)
        return tokens
    
    
    
class TokenizadorBPE:
    def __init__(self, vocab_size=150):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = [] 

    def train(self, corpus_text):
        
        words_freq = {}
        for word in corpus_text.split():
            words_freq[word] = words_freq.get(word, 0) + 1 

        splits = {}
        for word in words_freq.keys():
            split = list(word)
            self.vocab.update(split)
            splits[word] = split 
     
        while len(self.vocab) < self.vocab_size:
            pairs = {}
            for word, freq in words_freq.items():
                split = splits[word]
                for i in range(len(split) - 1):
                    pair = (split[i], split[i+1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get) 
            new_token = best_pair[0] + best_pair[1] 
            self.vocab.add(new_token)
            self.merge_rules.append((best_pair, new_token))

            for word in words_freq.keys():
                split = splits[word]
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i+1]) == best_pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                splits[word] = new_split

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            split = list(word) 
            
            for pair, new_token in self.merge_rules:
                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                        new_split.append(new_token)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                split = new_split
            tokens.extend(split)
        return tokens
    
    

if __name__ == "__main__":
    
    with open("test_sentences.txt", "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]

    print("=== RESULTADOS EN TEST_SENTENCES.TXT ===\n")

    print("--- Tokenización por Espacios ---")
    for line in test_lines:
        print(f"Input: '{line}' -> Tokens: {Token_espacios(line)}")
    
    print("\n--- Tokenización por Signos de Puntuación ---")
    for line in test_lines:
        print(f"Input: '{line}' -> Tokens: {Token_puntuacion(line)}")

    print("\n--- Tokenización por N-gramas (n=2) ---")
    for line in test_lines:
        print(f"Input: '{line}' -> Tokens: {Token_n_gramas(line, 2)}")

    print("\n--- Clasificador Supervisado ---")
    for line in test_lines:
        print(f"Input: '{line}' -> Tokens: {Token_clas_superv(line)}")



