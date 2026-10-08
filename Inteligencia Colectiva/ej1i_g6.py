import numpy as np
import matplotlib.pyplot as plt

# funcion q queremos minimizar
def funcion_objetivo(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# calcula la derivada de la funcion
def derivada_funcion(x):
    # en x = 0 tomamos la derivada como 0
    if x == 0:
        return 0

    # derivada de la funcion objetivo
    return (
        -np.sin(np.sqrt(np.abs(x)))
        - (np.sqrt(np.abs(x)) / 2)
        * np.cos(np.sqrt(np.abs(x)))
    )

# aplica gradiente descendente para buscar un minimo
def gradiente_descendente(x_inicial,tasa_aprendizaje,max_iteraciones):

    # comezamos desde el punto inicial
    x = x_inicial
    for _ in range(max_iteraciones):

        # calculamos la derivada en el punto actual
        gradiente = derivada_funcion(x)

        # nos movemos en sentido contrario al gradiente
        x_nuevo = (x - tasa_aprendizaje * gradiente)

        # mantenemos x dentro del intervalo permitido
        x_nuevo = np.clip(x_nuevo,xmin,xmax)

        # si el cambio es muy pequeño consideramos
        # q el metodo ya convergio
        if abs(x_nuevo - x) < 1e-8:
            break

        # actualizamos el valor de x
        x = x_nuevo
    # devolvemos la solucion encontrada y su valor en la funcion objetivo
    return x, funcion_objetivo(x)


# genera la poblacion inicial de cromosomas
def inicializar_poblacion(tam_poblacion,n_bits):

    # cada fila representa un individuo
    # cada columna representa un bit
    poblacion = np.random.randint(0,2,size=(tam_poblacion, n_bits))

    return poblacion


# convierte los cromosomas binarios a valores reales
def decodificar(poblacion,xmin,xmax):
    n_bits = poblacion.shape[1]
    # calcula el peso correspondiente a cada bit para un cromosoma de 4 bits seria [8, 4, 2, 1]
    potencias = 2 ** np.arange(n_bits - 1,-1,-1)

    # convierte cada cromosoma binario a decimal
    valores_enteros = poblacion @ potencias
    # transforma los valores enteros al intervalo [xmin, xmax]
    x = xmin + valores_enteros * (xmax - xmin) / (2**n_bits - 1)
    return x


# calcula el fitness de cada individuo
def calcular_fitness(x):

    f = funcion_objetivo(x)

    # queremos minimizar f(x), pero en la seleccion nos interesa q los individuos mejores tengan
    # un fitness mayor
    # usamos una normalizacion aproximada del rango de la funcion entre -430 y 430.
    # si f = -430, el fitness es 1.
    # si f = 430, el fitness es 0.
    # el 860 corresponde al ancho del intervalo:
    # 430 - (-430) = 860.
    fitness = (430 - f) / 860
    return fitness


# selecciona progenitores mediante competencias
def seleccion_competencias(poblacion,fitness,n_progenitores,k):

    progenitores = []
    for _ in range(n_progenitores):
        # elegimos k individuos al azar para competir
        participantes = np.random.choice(len(poblacion),size=k,replace=False)
        # gana el individuo con mayor fitness
        ganador = participantes[np.argmax(fitness[participantes])]
        # guardamos una copia del ganador
        progenitores.append(poblacion[ganador].copy())
    return np.array(progenitores)


# realiza una cruza simple entre dos padres
def cruza_simple(padre1,padre2):
    n_bits = len(padre1)
    # elegimos aleatoriamente el punto donde se van a intercambiar los genes
    punto = np.random.randint(1,n_bits)
    # cada hijo recibe una parte de cada padre
    hijo1 = np.concatenate([padre1[:punto],padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto],padre1[punto:]])
    return hijo1, hijo2


# aplica mutacion sobre los bits de un individuo
def mutacion(individuo,prob_mutacion):

    # trabajamos sobre una copia para no modificar directamente el individuo original
    individuo = individuo.copy()
    for i in range(len(individuo)):
        # con cierta probabilidad invertimos el bit
        if np.random.rand() < prob_mutacion:
            individuo[i] = 1 - individuo[i]

    return individuo


# parametros

# fijamos la semilla para poder repetir el experimento y obtener los mismos resultados
np.random.seed(42)
# limites del dominio de la funcion
xmin = -512
xmax = 512
# cantidad de bits utilizada para representar x
n_bits = 16
# cantidad de individuos de la poblacion
tam_poblacion = 49
# cantidad de progenitores q vamos a seleccionar
n_progenitores = 20
# cantidad de individuos q participan en cada competencia
k = 3
# probabilidad de mutacion de cada bit
prob_mutacion = 1 / n_bits
# cantidad maxima de generaciones
max_generaciones = 1000
# criterio de parada basado en el fitness
fitness_objetivo = 0.95

# criterio para comparar la velocidad de convergencia

# usamos el mismo nivel de calidad para comparar el algoritmo genetico con pso

# no reemplaza al criterio de parada del algoritmo
# solamente nos permite medir en q momento se alcanza
# una solucion suficientemente cercana al resultado final

umbral_convergencia = -418.9
# guardamos en q generacion se alcanza por primera vez este valor de la funcion
generacion_convergencia = None

# poblacion inicial
poblacion = inicializar_poblacion(tam_poblacion,n_bits)

# guardamos la evolucion del algoritmo
historial_fitness = []
historial_f = []
historial_x = []
historial_mejor = []
historial_promedio = []
historial_peor = []


# ciclo principal del algoritmo genetico
for generacion in range(max_generaciones):

    # convertimos los cromosomas binarios a valores de x
    x = decodificar(poblacion,xmin,xmax)

    # evaluamos la funcion objetivo
    f = funcion_objetivo(x)

    # calculamos el fitness de cada individuo
    fitness = calcular_fitness(x)

    # obtenemos el mejor, promedio y peor fitness
    fitness_mejor = np.max(fitness)
    fitness_promedio = np.mean(fitness)
    fitness_peor = np.min(fitness)

    # buscamos la posicion del mejor individuo
    indice_mejor = np.argmax(fitness)

    # obtenemos sus valores
    mejor_x = x[indice_mejor]
    mejor_f = f[indice_mejor]
    mejor_fitness = fitness[indice_mejor]

    # guardamos informacion sobre el mejor individuo
    historial_x.append(mejor_x)
    historial_f.append(mejor_f)
    historial_fitness.append(mejor_fitness)

    # guardamos estadisticas de toda la poblacion
    historial_mejor.append(fitness_mejor)
    historial_promedio.append(fitness_promedio)
    historial_peor.append(fitness_peor)

    # medir velocidad de convergencia
    # comprobamos si el mejor individuo de esta generacion alcanzo el mismo nivel de calidad q utilizaremos
    # para comparar con pso

    if (
        generacion_convergencia is None
        and mejor_f <= umbral_convergencia
    ):
        # sumamos 1 porque la generacion 0 representa la primera evaluacion de la poblacion
        generacion_convergencia = (generacion + 1)


    # mostramos informacion cada 50 generaciones
    if generacion % 50 == 0:

        print(
            f"generacion {generacion}: "
            f"x = {mejor_x:.5f}, "
            f"f(x) = {mejor_f:.5f}, "
            f"fitness = {mejor_fitness:.5f}"
        )


    # criterio de parada
    # se puede activar si queremos detenernos cuando encontremos una solucion suficientemente buena
    # lo dejamos comentado para observar las 1000 generaciones completas

    # if mejor_fitness >= fitness_objetivo:
    #
    #     print("\ncriterio de parada alcanzado.")
    #
    #     break

    # elitismo
    # guardamos el mejor individuo para asegurarnos
    # de q no se pierda en la siguiente generacion
    elitista = poblacion[indice_mejor].copy()


    # eliminamos el elitista antes de seleccionar de esta forma el elitista no puede ser seleccionado
    # nuevamente como progenitor en esta etapa

    poblacion_sin_elitista = np.delete(poblacion,indice_mejor,axis=0)
    fitness_sin_elitista = np.delete(fitness,indice_mejor)
    # seleccion de progenitores

    progenitores = seleccion_competencias(poblacion_sin_elitista,fitness_sin_elitista, n_progenitores,k)

    # generacion de hijos
    hijos = []

    # tenemos 49 individuos en total como uno se reserva para el elitista, necesitamos generar 48 hijos.
    # cada cruza genera 2 hijos,por lo q necesitamos 48 / 2 = 24 cruzas.

    for _ in range(24):
        # elegimos dos progenitores al azar del conjunto de progenitores seleccionados
        indice_padre1 = np.random.randint(0,n_progenitores)
        indice_padre2 = np.random.randint(0,n_progenitores)

        padre1 = progenitores[indice_padre1]
        padre2 = progenitores[indice_padre2]

        # realizamos la cruza
        hijo1, hijo2 = cruza_simple(padre1,padre2)


        # aplicamos mutacion a los dos hijos
        hijo1 = mutacion(hijo1,prob_mutacion)
        hijo2 = mutacion(hijo2,prob_mutacion)

        # guardamos los hijos
        hijos.append(hijo1)
        hijos.append(hijo2)

    hijos = np.array(hijos)


    # nueva poblacion

    # incorporamos el elitista junto con los 48 hijos para volver a tener 49 individuos

    poblacion = np.vstack([elitista, hijos])

# resultado final del algoritmo genetico


print()
print("==============================")
print("resultado final")
print(f"generacion: {generacion}")
print(f"x encontrado: {mejor_x:.8f}")
print(f"f(x): {mejor_f:.8f}")
print(f"fitness: {mejor_fitness:.8f}")



# velocidad de convergencia
print()
print("==============================")
print("velocidad de convergencia")
print("==============================")


if generacion_convergencia is not None:
    print(
        f"umbral utilizado: "
        f"f(x) <= {umbral_convergencia}"
    )
    print(
        f"generaciones necesarias: "
        f"{generacion_convergencia}"
    )
else:
    print(
        "el umbral de convergencia "
        "no fue alcanzado"
    )


# graficamos la evolucion de la poblacion
plt.plot(historial_mejor,label="mejor")

plt.plot(historial_promedio, label="promedio")

plt.plot(historial_peor,label="peor")
plt.xlabel("generacion")
plt.ylabel("fitness")
plt.title("evolucion de la poblacion")
plt.legend()
plt.grid()
plt.show()

# gradiente descendente
print()
print("==============================")
print("gradiente descendente")
# elegimos un punto inicial para comenzar la busqueda
x_inicial = 350
# define cuanto avanza el metodo en cada iteracion
tasa_aprendizaje = 0.1
# cantidad maxima de iteraciones
max_iteraciones = 10000
# ejecutamos gradiente descendente
x_gradiente, f_gradiente = gradiente_descendente(x_inicial,tasa_aprendizaje,max_iteraciones)
# mostramos el resultado obtenido
print(f"x inicial: {x_inicial}")
print(f"x encontrado: {x_gradiente:.8f}")
print(f"f(x): {f_gradiente:.8f}")