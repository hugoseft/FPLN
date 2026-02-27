import matplotlib.pyplot as plt
from Segmentacion import *

def cargar_oraciones(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def analizar_evolucion():
    oraciones = cargar_oraciones("majesty_speeches.txt")
    corpus_completo = " ".join(oraciones)
    
    # --- Configuración de Tokenizadores ---
    # Para WP y BPE entrenamos primero con el corpus completo (límite 3000)
    print("Entrenando WordPiece y BPE con majesty_speeches.txt...")
    wp = TokenizadorWordPiece(vocab_size=3000)
    wp.train(corpus_completo)
    
    bpe = TokenizadorBPE(vocab_size=3000)
    bpe.train(corpus_completo)
    
    # --- Estructuras para el seguimiento ---
    metodos = {
        "Espacios": lambda x: Token_espacios(x),
        "Puntuación": lambda x: Token_puntuacion(x),
        "N-gramas (n=2)": lambda x: Token_n_gramas(x, 2),
        "WordPiece": lambda x: wp.tokenize(x),
        "BPE": lambda x: bpe.tokenize(x)
    }
    
    resultados = {nombre: [] for nombre in metodos}
    vocabularios_unicos = {nombre: set() for nombre in metodos}
    eje_x = list(range(1, len(oraciones) + 1))

    print("Procesando oraciones para el análisis visual...")
    for i, oracion in enumerate(oraciones):
        for nombre, func in metodos.items():
            # Obtenemos los tokens de la oración actual
            tokens = func(oracion)
            # Los añadimos al set de tokens únicos vistos hasta ahora
            vocabularios_unicos[nombre].update(tokens)
            # Guardamos el tamaño actual del vocabulario
            resultados[nombre].append(len(vocabularios_unicos[nombre]))

    # --- Creación de la Representación Visual ---
    plt.figure(figsize=(12, 7))
    for nombre, valores in resultados.items():
        plt.plot(eje_x, valores, label=nombre)

    plt.title("Evolución del Tamaño del Vocabulario (majesty_speeches.txt)")
    plt.xlabel("Número de Oraciones Procesadas")
    plt.ylabel("Número de Tokens Únicos (Vocabulario)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Guardar y mostrar
    plt.savefig("evolucion_vocabulario.png")
    print("Gráfico generado: evolucion_vocabulario.png")
    plt.show()

if __name__ == "__main__":
    analizar_evolucion()