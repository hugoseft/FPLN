import pickle
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Función para extraer las 4 características que pide el enunciado
def extraer_caracteristicas(caracter, siguiente_caracter):
    es_puntuacion = caracter in string.punctuation
    es_numero = caracter.isdigit()
    
    return {
        'char': caracter,            # La propia letra
        'next_char': siguiente_caracter, # La letra siguiente
        'is_num': es_numero,         # Booleano: ¿es número?
        'is_punct': es_puntuacion    # Booleano: ¿es puntuación?
    }

def entrenar_modelo():
    features = [] # Lista de características (X)
    labels = []   # Lista de respuestas correctas (Y) - 1 si corta, 0 si sigue
    
    # Leemos el fichero de entrenamiento
    # IMPORTANTE: El fichero debe tener espacios donde van los cortes.
    # Ej: "Hola , mundo !"
    with open("training_sentences.txt", "r", encoding="utf-8") as f:
        lineas = f.readlines()

    for linea in lineas:
        palabras = linea.strip().split() # Separamos por espacios (la "verdad")
        
        # Reconstruimos la frase para ir letra por letra
        full_text = "".join(palabras) 
        idx_global = 0
        
        for palabra in palabras:
            for i, char in enumerate(palabra):
                # Miramos el siguiente caracter (si existe, sino es un marcador de fin)
                next_char = palabra[i+1] if i+1 < len(palabra) else ""
                
                # Si estamos en el ultimo caracter de la palabra, el siguiente real
                # en el texto completo sería el inicio de la siguiente palabra.
                # Para simplificar, usamos lógica de reconstrucción:
                
                try:
                    siguiente_real = full_text[idx_global + 1]
                except IndexError:
                    siguiente_real = "EOF" # Fin del texto

                # Extraemos características
                feats = extraer_caracteristicas(char, siguiente_real)
                features.append(feats)
                
                # LA ETIQUETA (LABEL):
                # Si es el último caracter de la palabra, es una frontera (1).
                # Si no, no lo es (0).
                if i == len(palabra) - 1:
                    labels.append(1) # AQUÍ CORTAMOS
                else:
                    labels.append(0) # AQUÍ SEGUIMOS
                
                idx_global += 1

    # 2. Convertir a vectores One-Hot (DictVectorizer lo hace automático)
    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(features)
    y_train = labels

    # 3. Entrenar Regresión Logística
    print("Entrenando modelo... (esto puede tardar un poco)")
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    
    # 4. Guardar el modelo y el vectorizador
    # Guardamos el "cerebro" para no re-entrenar siempre
    with open("modelo_token.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("vectorizador.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("¡Entrenamiento completado! Archivos .pkl generados.")

if __name__ == "__main__":
    entrenar_modelo()