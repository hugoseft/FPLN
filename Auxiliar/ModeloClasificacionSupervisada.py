import pickle
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Función para extraer características de un carácter dado y su siguiente carácter
def extraer_caracteristicas(caracter, siguiente_caracter):
    es_puntuacion = caracter in string.punctuation
    es_numero = caracter.isdigit()
    
    return {
        'char': caracter,                # Letra
        'next_char': siguiente_caracter, # Letra siguiente
        'is_num': es_numero,             # Booleano por si es número
        'is_punct': es_puntuacion        # Booleano por si es puntuación
    }


# Función para entrenar el modelo supervisado
def entrenar_modelo(entrenamiento_path="training_sentences.txt", modelo_path="modelo_token.pkl", vectorizer_path="vectorizador.pkl"):
    features = [] # Lista de características (X)
    labels = []   # Lista de respuestas correctas (Y) - 1 si corta, 0 si sigue
    
    with open(entrenamiento_path, "r", encoding="utf-8") as f:
        lineas = f.readlines()

    for linea in lineas:
        palabras = linea.strip().split()
        
        full_text = "".join(palabras) 
        idx_global = 0
        
        for palabra in palabras:
            for i, char in enumerate(palabra):
                
                next_char = palabra[i+1] if i+1 < len(palabra) else ""
                
                try:
                    siguiente_real = full_text[idx_global + 1]
                except IndexError:
                    siguiente_real = "EOF" 

                feats = extraer_caracteristicas(char, siguiente_real)
                features.append(feats)
                
                if i == len(palabra) - 1:
                    labels.append(1) 
                else:
                    labels.append(0)
                
                idx_global += 1

    # Convertir a vectores One-Hot (con DictVectorizer)
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(features)
    y_train = labels

    # Entrenar Regresión Logística
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Guardar el modelo y el vectorizador
    with open(modelo_path, "wb") as f:
        pickle.dump(clf, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        

if __name__ == "__main__":

    entrenar_modelo()
