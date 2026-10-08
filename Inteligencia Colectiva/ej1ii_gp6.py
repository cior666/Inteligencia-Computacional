import numpy as np
import matplotlib.pyplot as plt

# calcula la funcion objetivo del ejercicio
def funcion_objetivo(x, y):
    # agrupamos x e y en una sola variable
    r = x**2 + y**2
    # devolvemos el valor de la funcion
    return r**0.25 * (np.sin(50 * r**0.1)**2 + 1)

# calcula el gradiente de la funcion respecto de x e y
def gradiente_funcion(x, y):
    # calculamos r = x^2 + y^2
    r = x**2 + y**2
    # en el origen evitamos calcular potencias negativas q podrian generar problemas numericos
    if r == 0:
        return 0, 0
    # calculamos la primera parte de la derivada
    termino1 = (0.25 * r**(-0.75) * (np.sin(50 * r**0.1)**2 + 1))
    # calculamos la segunda parte de la derivada
    termino2 = (10 * r**(-0.65) * np.sin(50 * r**0.1) * np.cos(50 * r**0.1))
    # sumamos los dos terminos para obtener df/dr
    df_dr = termino1 + termino2
    # aplicamos la regla de la cadena para obtener
    # las derivadas respecto de x e y
    df_dx = 2 * x * df_dr
    df_dy = 2 * y * df_dr
    return df_dx, df_dy

# aplica gradiente descendente para buscar un minimo
def gradiente_descendente(x_inicial,y_inicial,tasa_aprendizaje,max_iteraciones):
    # comenzamos desde el punto inicial elegido
    x = x_inicial
    y = y_inicial
    for _ in range(max_iteraciones):
        # calculamos el gradiente en el punto actual
        grad_x, grad_y = gradiente_funcion(x, y)
        # avanzamos en sentido contrario al gradiente
        # para intentar disminuir el valor de la funcion
        x_nuevo = x - tasa_aprendizaje * grad_x
        y_nuevo = y - tasa_aprendizaje * grad_y
        # mantenemos el punto dentro del dominio
        x_nuevo = np.clip(x_nuevo,xmin,xmax)
        y_nuevo = np.clip(y_nuevo,xmin,xmax)
        # si x e y practicamente no cambian, consideramos q el metodo convergio
        if (abs(x_nuevo - x) < 1e-8 and abs(y_nuevo - y) < 1e-8):
            break
        # actualizamos el punto para la siguiente iteracion
        x = x_nuevo
        y = y_nuevo
    # devolvemos el punto encontrado y su valor en la funcion objetivo
    return x, y, funcion_objetivo(x, y)

# genera la poblacion inicial del algoritmo genetico
def inicializar_poblacion(tam_poblacion,n_bits):
    # cada fila representa un individuo
    # cada columna representa un bit
    poblacion = np.random.randint(0,2,size=(tam_poblacion, n_bits))
    return poblacion

# transforma los cromosomas binarios en valores reales
def decodificar(poblacion,xmin,xmax):
    # la mitad de los bits representa x y la otra mitad representa y
    n_bits = poblacion.shape[1] // 2
    # calculamos el peso de cada bit
    potencias = 2 ** np.arange(n_bits - 1,-1,-1)
    # convertimos los primeros bits de cada individuo a un valor entero para obtener x
    valores_x = (poblacion[:, :n_bits] @ potencias)
    # transformamos el valor entero al intervalo [xmin, xmax]
    x = xmin + valores_x * (xmax - xmin) / (2**n_bits - 1)
    # convertimos los bits restantes a un valor entero para obtener y
    valores_y = (poblacion[:, n_bits:] @ potencias)
    # transformamos el valor entero al intervalo [xmin, xmax]
    y = xmin + valores_y * (xmax - xmin) / (2**n_bits - 1)
    return x, y

# calcula el fitness de cada individuo
def calcular_fitness(x, y):
    # calculamos el valor de la funcion objetivo
    f = funcion_objetivo(x, y)
    # queremos minimizar f, pero en el algoritmo genetico nos interesa q las mejores soluciones tengan un fitness mayor
    # por eso usamos 1 / (1 + f)
    # cuando f se acerca a 0, el fitness se acerca a 1.
    fitness = 1 / (1 + f)
    return fitness

# selecciona progenitores mediante competencias
def seleccion_competencias(poblacion,fitness,n_progenitores,k):
    progenitores = []
    for _ in range(n_progenitores):
        # elegimos k individuos al azar para competir
        participantes = np.random.choice(len(poblacion), size=k,replace=False)
        # gana el individuo con mayor fitness
        ganador = participantes[np.argmax(fitness[participantes])]
        # guardamos una copia del ganador
        progenitores.append(poblacion[ganador].copy())
    return np.array(progenitores)

# realiza una cruza simple entre dos progenitores
def cruza_simple(padre1, padre2):
    n_bits = len(padre1)
    # elegimos aleatoriamente el punto
    # donde se van a intercambiar los bits
    punto = np.random.randint(1,n_bits)
    #cada hijo recibe una parte de cada padre
    hijo1 = np.concatenate([padre1[:punto],padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto],padre1[punto:]])
    return hijo1, hijo2

# aplica mutacion sobre los bits de un individuo
def mutacion(individuo,prob_mutacion):
    # hacemos una copia para no modificardirectamente el individuo original
    individuo = individuo.copy()
    for i in range(len(individuo)):
        # con cierta probabilidad invertimos el bit
        if np.random.rand() < prob_mutacion:
            individuo[i] = 1 - individuo[i]
    return individuo

# parametros

# limites del dominio para x e y
xmin = -100
xmax = 100

# usamos 16 bits para representar x
n_bits_x = 16

# usamos 16 bits para representar y
n_bits_y = 16

# cada cromosoma tiene 32 bits en total
n_bits = n_bits_x + n_bits_y

# cantidad de individuos de la poblacion
tam_poblacion = 99

# cantidad de progenitores q vamos a seleccionar
n_progenitores = 20

# cantidad de individuos q participan en cada competencia
k = 3

# probabilidad de mutacion de cada bit
prob_mutacion = 1 / n_bits

# cantidad maxima de generaciones
max_generaciones = 1000

# valor de f a partir del cual consideramos q encontramos una solucion suficientemente buena
f_objetivo = 0.05

# valor de referencia para medir la velocidad de convergencia usamos un umbral distinto al criterio de parada
# para poder comparar cuantos pasos necesita cada metodo
umbral_convergencia = 0.1

# guardamos la primera generacion en la q se alcanza el umbral
generacion_convergencia = None

# poblacion inicial
poblacion = inicializar_poblacion(tam_poblacion,n_bits)

# guardamos la evolucion de la poblacion
historial_mejor = []
historial_promedio = []
historial_peor = []
historial_f = []
historial_x = []
historial_y = []

# ciclo principal 
for generacion in range(max_generaciones):
    # convertimos los cromosomas binarios en valores reales de x e y
    x, y = decodificar(poblacion,xmin,xmax)
    # calculamos el valor de la funcion para todos los individuos
    f = funcion_objetivo(x, y)
    # calculamos el fitness de todos los individuos
    fitness = calcular_fitness(x, y)
    # buscamos la posicion del individuo con mayor fitness
    indice_mejor = np.argmax(fitness)

    # obtenemos los datos del mejor individuo
    mejor_x = x[indice_mejor]
    mejor_y = y[indice_mejor]
    mejor_f = f[indice_mejor]
    mejor_fitness = fitness[indice_mejor]

    # guardamos los datos del mejor individuo para poder analizarlos despues
    historial_x.append(mejor_x)
    historial_y.append(mejor_y)
    historial_f.append(mejor_f)

    # velocidad de convergencia
    # verificamos si el mejor valor encontrado ya alcanzo el mismo nivel de calidad usado para comparar los metodos
    if (generacion_convergencia is None and mejor_f <= umbral_convergencia):
        # guardamos la cantidad de generaciones necesarias
        generacion_convergencia = generacion + 1

    # guardamos el mejor, promedio y peor fitness de cada generacion
    historial_mejor.append(np.max(fitness))
    historial_promedio.append(np.mean(fitness))
    historial_peor.append(np.min(fitness))

    # mostramos informacion cada 50 generaciones
    if generacion % 50 == 0:
        print(
            f"generacion {generacion}: "
            f"x = {mejor_x:.5f}, "
            f"y = {mejor_y:.5f}, "
            f"f(x,y) = {mejor_f:.5f}, "
            f"fitness = {mejor_fitness:.5f}"
        )

    # criterio de parada
    # si encontramos una solucion con f <= 0.05,consideramos q el objetivo fue alcanzado
    if mejor_f <= f_objetivo:
        print("\ncriterio de parada alcanzado.")
        break


    # elitismo
    # guardamos el mejor individuo para asegurarnos de q no se pierda en la siguiente generacion
    elitista = poblacion[indice_mejor].copy()

    # quitamos temporalmente el elitista de la poblacion antes de realizar la seleccion
    poblacion_sin_elitista = np.delete(poblacion,indice_mejor,axis=0)
    fitness_sin_elitista = np.delete(fitness,indice_mejor)

    # seleccion de progenitores
    # seleccionamos 20 progenitores mediante competencias
    progenitores = seleccion_competencias(poblacion_sin_elitista,fitness_sin_elitista,n_progenitores,k)

    # generacion de hijos
    hijos = []

    # tenemos 99 individuos en la poblacion. uno queda reservado para el elitista
    # por lo tanto necesitamos generar: 99 - 1 = 98 hijos como cada cruza genera 2 hijos:
    # 98 / 2 = 49 cruzas

    for _ in range(49):
        # elegimos dos progenitores al azar del conjunto de 20 seleccionados
        indice_padre1 = np.random.randint(0,n_progenitores)
        indice_padre2 = np.random.randint(0,n_progenitores)
        padre1 = progenitores[indice_padre1]
        padre2 = progenitores[indice_padre2]

        # realizamos la cruza entre los dos padres
        hijo1, hijo2 = cruza_simple(padre1,padre2)

        # aplicamos mutacion a cada hijo
        hijo1 = mutacion(hijo1,prob_mutacion)
        hijo2 = mutacion(hijo2,prob_mutacion)

        # guardamos los hijos generados
        hijos.append(hijo1)
        hijos.append(hijo2)

    hijos = np.array(hijos)
    # nueva poblacion
    # formamos la nueva poblacion con:
    # 1 elitista + 98 hijos = 99 individuos
    poblacion = np.vstack([elitista,hijos])

# resultado final del algoritmo genetico
print()
print("==============================")
print("resultado final")
print("==============================")
print(f"generacion: {generacion}")
print(f"x encontrado: {mejor_x:.8f}")
print(f"y encontrado: {mejor_y:.8f}")
print(f"f(x,y): {mejor_f:.8f}")
print(f"fitness: {mejor_fitness:.8f}")


print()
print("==============================")
print("velocidad de convergencia")
print("==============================")
print(f"umbral utilizado: f(x,y) <= {umbral_convergencia}")
print(f"generaciones necesarias: {generacion_convergencia}")







# grafica de la evolucion
# mostramos como fueron cambiando el mejor, promedio y peor fitness de la poblacion
plt.plot( historial_mejor,label="mejor")
plt.plot(historial_promedio,label="promedio")
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
print("==============================")
# elegimos el punto desde donde comienza la busqueda del gradiente
x_inicial = 0.5
y_inicial = 0.5
# determina cuanto cambia x e y en cada iteracion
tasa_aprendizaje = 0.1
# cantidad maxima de iteraciones del metodo
max_iteraciones = 10000
# ejecutamos gradiente descendente
x_gradiente, y_gradiente, f_gradiente = (gradiente_descendente(x_inicial,y_inicial,tasa_aprendizaje,max_iteraciones))

# mostramos el resultado obtenido por gradiente
print(f"x inicial: {x_inicial}")
print(f"y inicial: {y_inicial}")
print(f"x encontrado: {x_gradiente:.8f}")
print(f"y encontrado: {y_gradiente:.8f}")
print(f"f(x,y): {f_gradiente:.8f}")