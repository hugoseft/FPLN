#TODO
import regex as re 
import pickle
import string
from Auxiliar.ModeloClasificacionSupervisada import extraer_caracteristicas, entrenar_modelo

#Funcion para tokenizar por espacios
def Token_espacios(texto):
    tokens = texto.split(" ")
    return tokens

#Funcion para tokenizar por signos de puntuación
def Token_puntuacion(texto):
    pattern = r"\w+|[^\w\s]|\p{So}"
    tokens = re.findall(pattern, texto, flags=re.UNICODE)
    return tokens

#Funcion para tokenizar por N-gramas
def Token_n_gramas(texto, n, tokenizacion_func=Token_espacios):
    tokens = tokenizacion_func(texto)
    n_gramas = []
    for i in range(len(tokens) - n + 1):
        n_grama = tokens[i:i+n]
        n_grama = " ".join(n_grama)
        n_gramas.append(n_grama)
    return n_gramas

# Función para tokenizar usando un clasificador supervisado
def Token_clas_superv(texto):
    
    #Cargar el modelo ya entrenado
    tokens = []
    
    try:
        with open("Auxiliar/modelo_token.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("Auxiliar/vectorizador.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        return ["Error: Ejecuta primero entrenar.py para generar el modelo"]

    texto_sin_espacios = texto.replace(" ","")
    features_list = []

    #Inferencia: Extraemos características de cada carácter y su siguiente carácter
    for i, char in enumerate(texto_sin_espacios):
        next_c = texto_sin_espacios[i+1] if i+1 < len(texto_sin_espacios) else "EOF"
        features_list.append(extraer_caracteristicas(char, next_c))
    
    X_test = vectorizer.transform(features_list) 
    predicciones = clf.predict(X_test)
    
    # Reconstruimos los tokens a partir de las predicciones
    palabra_actual = ""
    for i, char in enumerate(texto_sin_espacios):
        palabra_actual += char
        
        if predicciones[i] == 1:
            tokens.append(palabra_actual)
            palabra_actual = ""
            
    if palabra_actual:
        tokens.append(palabra_actual)
        
    return tokens


# Implementación de WordPiece
class TokenizadorWordPiece:

    def __init__(self, vocab_size=150):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.splits = {}

    # Método de entrenamiento para WordPiece
    def train(self, corpus_text):
       
       #Contamos frecuencias de palabras y añadimos el token [UNK] 
        words_freq = {}
        for word in corpus_text.split():
            words_freq[word] = words_freq.get(word, 0) + 1

        self.vocab.add("[UNK]") 
        
       # Dividimos cada palabra en letras, poniéndole "##" a las letras que no van al inicio
        for word in words_freq.keys():
            split = []
            for i, char in enumerate(word):
                token = char if i == 0 else "##" + char 
                split.append(token)
                self.vocab.add(token)
            self.splits[word] = split 

    # Bucle principal: Buscamos el par de tokens más frecuente y los fusionamos
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

            # Actualizamos todas las palabras para que usen esta nueva fusión
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

    # Método de tokenización para WordPiece
    def tokenize(self, text):
        tokens = []
        for word in text.split(): 
            if word in self.vocab:
                tokens.append(word)
                continue
            
            sub_tokens = []
            start = 0
            is_unknown = False
            
            # Intentamos encajar el trozo más largo posible que ya exista en el vocabulario
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
                
                # Si no encontramos ningún trozo válido, ponemos [UNK]
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
    
    
# Implementación de BPE
class TokenizadorBPE:

    def __init__(self, vocab_size=150):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = [] 

    # Método de entrenamiento para BPE
    def train(self, corpus_text):
        
        # Contamos las palabras y las rompemos en letras sueltas
        words_freq = {}
        for word in corpus_text.split():
            words_freq[word] = words_freq.get(word, 0) + 1 

        splits = {}
        for word in words_freq.keys():
            split = list(word)
            self.vocab.update(split)
            splits[word] = split 

        # Bucle principal: Encontrar el par de caracteres más común para unirlos
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
            # Guardamos la regla de fusión porque el ORDEN en que las aprendemos importa
            self.merge_rules.append((best_pair, new_token))

            # Actualizamos los splits aplicando la nueva fusión recién aprendida
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

    # Método de tokenización para BPE
    def tokenize(self, text):
        tokens = []
        for word in text.split():
            split = list(word) # Rompemos la palabra a tokenizar en letras base
            
            # Aplicamos todas las reglas de fusión exactamente en el mismo orden que las aprendimos
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
    
# Función para mostrar por pantalla y guardar en un archivo de texto al mismo tiempo
def print_y_guardar(texto, archivo):
    print(texto) 
    archivo.write(str(texto) + "\n") 

if __name__ == "__main__":
    
    # Espacios, Puntuación, N-gramas, y Supervisado
    with open("Resultado/resultados_ejecucion.txt", "w", encoding="utf-8") as f_out:
        
        with open("Auxiliar/test_sentences.txt", "r", encoding="utf-8") as f:
            test_lines = [line.strip() for line in f if line.strip()]

        print_y_guardar("=== RESULTADOS EN TEST_SENTENCES.TXT ===", f_out)
        print_y_guardar("", f_out) 

        print_y_guardar("--- Tokenización por Espacios ---", f_out)
        for line in test_lines:
            print_y_guardar(f"Input: '{line}' -> Tokens: {Token_espacios(line)}", f_out)
        
        print_y_guardar("\n--- Tokenización por Signos de Puntuación ---", f_out)
        for line in test_lines:
            print_y_guardar(f"Input: '{line}' -> Tokens: {Token_puntuacion(line)}", f_out)

        print_y_guardar("\n--- Tokenización por N-gramas (n=2) ---", f_out)
        for line in test_lines:
            print_y_guardar(f"Input: '{line}' -> Tokens: {Token_n_gramas(line, 2)}", f_out)

        print_y_guardar("\n--- Clasificador Supervisado ---", f_out)
        entrenar_modelo("Auxiliar/training_sentences.txt", "Auxiliar/modelo_token.pkl", "Auxiliar/vectorizador.pkl")
        for line in test_lines:
            print_y_guardar(f"Input: '{line}' -> Tokens: {Token_clas_superv(line)}", f_out)
        

        # WordPiece y BPE con vocabulario de 100, 150 y 200
        with open("Auxiliar/majesty_speeches.txt", "r", encoding="utf-8") as f:
            train_lines = [line.strip() for line in f if line.strip()]

        corpus_train_completo = " ".join(train_lines)

        tamanos_vocabulario = [100, 150, 200]

        for size in tamanos_vocabulario:
            print_y_guardar(f"\n=========================================", f_out)
            print_y_guardar(f"   EVALUANDO TAMAÑO DE VOCABULARIO: {size}", f_out)
            print_y_guardar(f"=========================================", f_out)
            
            # WordPiece
            print_y_guardar(f"\n--- WordPiece (Vocab: {size}) ---", f_out)
            wp_tokenizer = TokenizadorWordPiece(vocab_size=size)
            wp_tokenizer.train(corpus_train_completo)
            print_y_guardar(f"Vocabulario WordPiece ({size}):\n{wp_tokenizer.vocab}\n", f_out)
            
            for line in test_lines:
                print_y_guardar(f"Input: '{line}' -> Tokens: {wp_tokenizer.tokenize(line)}", f_out)

            # BPE
            print_y_guardar(f"\n--- BPE (Vocab: {size}) ---", f_out)
            bpe_tokenizer = TokenizadorBPE(vocab_size=size)
            bpe_tokenizer.train(corpus_train_completo)
            print_y_guardar(f"Vocabulario BPE ({size}):\n{bpe_tokenizer.vocab}\n", f_out)
            
            for line in test_lines:
                print_y_guardar(f"Input: '{line}' -> Tokens: {bpe_tokenizer.tokenize(line)}", f_out)



