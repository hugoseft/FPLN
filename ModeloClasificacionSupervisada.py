import pickle
import string
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def extraer_caracteristicas(caracter, siguiente_caracter):
    es_puntuacion = caracter in string.punctuation
    es_numero = caracter.isdigit()
    
    return {
        'char': caracter,          
        'next_char': siguiente_caracter, 
        'is_num': es_numero,        
        'is_punct': es_puntuacion    
    }

def entrenar_modelo():
    features = [] 
    labels = []  
    
   
    with open("training_sentences.txt", "r", encoding="utf-8") as f:
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

    vectorizer = DictVectorizer(sparse=False)
    X_train = vectorizer.fit_transform(features)
    y_train = labels

    print("Entrenando modelo... (esto puede tardar un poco)")
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X_train, y_train)
    

    with open("modelo_token.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("vectorizador.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("¡Entrenamiento completado! Archivos .pkl generados.")

if __name__ == "__main__":
    entrenar_modelo()
