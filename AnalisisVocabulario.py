import matplotlib.pyplot as plt
import pickle
from Segmentacion import *

# Optimización para el método supervisado en el análisis
def cargar_modelo_supervisado():
    try:
        with open("modelo_token.pkl", "rb") as f:
            clf = pickle.load(f)
        with open("vectorizador.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return clf, vectorizer
    except FileNotFoundError:
        print("Error: No se encuentran los archivos .pkl. Entrena el modelo primero.")
        return None, None

def analizar_evolucion():

    with open("majesty_speeches.txt", "r", encoding="utf-8") as f:
        oraciones = [line.strip() for line in f if line.strip()]
    
    corpus_completo = " ".join(oraciones)
    
    print("Entrenando WordPiece y BPE (Vocab: 3000)...")
    wp = TokenizadorWordPiece(vocab_size=3000)
    wp.train(corpus_completo)
    
    bpe = TokenizadorBPE(vocab_size=3000)
    bpe.train(corpus_completo)
    
    clf, vectorizer = cargar_modelo_supervisado()
    
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

    print("Procesando oraciones y calculando vocabularios...")
    for i, oracion in enumerate(oraciones):
        eje_x.append(i + 1)
        for nombre, func in metodos.items():
            tokens = func(oracion)
            vocabularios_vistos[nombre].update(tokens)
            resultados[nombre].append(len(vocabularios_vistos[nombre]))


    ### GRAFICAMOS TODO

    plt.figure(figsize=(12, 7))
    for nombre, valores in resultados.items():
        plt.plot(eje_x, valores, label=nombre)

    plt.title("Evolución del Vocabulario por Método (Majesty Speeches)")
    plt.xlabel("Número de Oraciones")
    plt.ylabel("Tokens Únicos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparativa_vocabulario.png")
    print("Gráfica guardada como 'comparativa_vocabulario.png'")
    plt.show()

if __name__ == "__main__":
    analizar_evolucion()