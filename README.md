# FPLN



Se pide también realizar
una discusión sobre las diferencias observadas. (entre la salida de entrenamiento y prueba)


![alt text](image.png) (IMAGEN GRÁFICA)

ANÁLISIS DE LA GRÁFICA

1. Eficiencia de las Subpalabras (BPE y WordPiece)Estabilidad: Estos métodos son los más eficientes, ya que el tamaño del vocabulario se estabiliza rápidamente al alcanzar el límite preestablecido (en este caso, 3000 tokens).+2Reutilización: Al dividir las palabras en unidades léxicas más pequeñas (prefijos, raíces, sufijos), logran representar todo el texto sin necesidad de crear tokens nuevos constantemente para variaciones de una misma palabra.+1

2. La Complejidad de los N-gramasCrecimiento Masivo: Es el método con mayor pendiente en el gráfico, debido a que genera todas las combinaciones posibles de $n$ palabras contiguas.+1Saturación: Esto demuestra que, aunque capturan contexto local, requieren un espacio de memoria mucho mayor al generar miles de secuencias únicas que rara vez se repiten de la misma forma.

3. El Comportamiento del Método SupervisadoDecisiones de Probabilidad: Al clasificar cada carácter para decidir si es una frontera de token o no, el modelo es propenso a errores de predicción según lo aprendido en el entrenamiento.+2Resultado en Gráfica: Su crecimiento es superior al de los métodos clásicos (Espacios) porque cualquier error en la detección de una frontera genera fragmentos de palabras que el sistema registra como tokens únicos nuevos.+1

4. Comparativa de Métodos Clásicos (Espacios vs. Puntuación)Tokenización por Espacios: Es el modelo más básico y el vocabulario crece rápido porque incluye los signos de puntuación pegados a las palabras (ej. "casa" y "casa." se cuentan como dos tokens distintos).+1Tokenización por Puntuación: Es más limpia y compacta. Al tratar los símbolos y emojis como unidades independientes, reduce el número de tokens únicos totales al normalizar las palabras.+1








TAREAS PENDIENTES


- JUNTAR TODO EL CÓDIGO Y HACER QUE LAS EJECUCIONES SE GUARDEN Y SE ACTUALICEN EN UN ARCHIVO (EJEMPLO "resulados")
