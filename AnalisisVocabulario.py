import matplotlib.pyplot as plt
import pickle
from Segmentacion import *

# Funcion para cargar el modelo supervisado entrenado
def cargar_modelo_supervisado():
    try:
        with open("Auxiliar/modelo_token.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("Auxiliar/vectorizador.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return clf, vectorizer
    except FileNotFoundError:
        print("Error: No se encuentran los archivos .pkl. Entrena el modelo primero.")
        return None, None

#Funcion para comparar el vocabulario de los métodos de tokenización
def analizar_evolucion():

    # Leemos todo el texto y lo juntamos para tener un buen corpus
    with open("Auxiliar/majesty_speeches.txt", "r", encoding="utf-8") as f:
        oraciones = [line.strip() for line in f if line.strip()]
    
    corpus_completo = " ".join(oraciones)

    # Entrenamos previamente los modelos WordPiece y BPE
    print("Entrenando WordPiece y BPE (Vocab: 3000)...")
    wp = TokenizadorWordPiece(vocab_size=3000)
    wp.train(corpus_completo)
    
    bpe = TokenizadorBPE(vocab_size=3000)
    bpe.train(corpus_completo)
    
    clf, vectorizer = cargar_modelo_supervisado()
    
    # Metemos todos los métodos de tokenización en un diccionario para evaluarlos juntos
    metodos = {
        "Espacios": lambda x: Token_espacios(x),
        "Puntuación": lambda x: Token_puntuacion(x),
        "N-gramas (n=2)": lambda x: Token_n_gramas(x, 2),
        "WordPiece (3000)": lambda x: wp.tokenize(x),
        "BPE (3000)": lambda x: bpe.tokenize(x)
    }
    
    if clf and vectorizer:
        metodos["Supervisado"] = lambda x: Token_clas_superv(x)

    resultados = {nombre: [] for nombre in metodos}
    vocabularios_vistos = {nombre: set() for nombre in metodos}
    eje_x = []

    # Leemos frase por frase y contamos cuántos tokens NUEVOS descubre cada método
    print("Procesando oraciones y calculando vocabularios...")
    for i, oracion in enumerate(oraciones):
        eje_x.append(i + 1)
        for nombre, func in metodos.items():
            tokens = func(oracion)
            vocabularios_vistos[nombre].update(tokens)
            resultados[nombre].append(len(vocabularios_vistos[nombre]))


    # Dibujamos la gráfica para ver la evolución y la guardamos

    plt.figure(figsize=(12, 7))
    for nombre, valores in resultados.items():
        plt.plot(eje_x, valores, label=nombre)

    plt.title("Evolución del Vocabulario por Método (Majesty Speeches)")
    plt.xlabel("Número de Oraciones")
    plt.ylabel("Tokens Únicos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("Resultado/comparativa_vocabulario.png")
    print("Gráfica guardada como 'comparativa_vocabulario.png'")
    plt.show()

if __name__ == "__main__":
    analizar_evolucion()