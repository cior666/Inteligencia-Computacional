import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

# cargar los datos
# cargamos los archivos de entrenamiento y test usamos r antes de la ruta para q python interprete correctamente las barras de windows

train = pd.read_csv(
    r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Inteligencia Colectiva\leukemia_train.csv",
    header=None
)
test = pd.read_csv(
    r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Inteligencia Colectiva\leukemia_test.csv",
    header=None
)

# las primeras columnas corresponden a las caracteristicas la ultima columna corresponde a la clase
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values.astype(int)
X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values.astype(int)

# mostramos las dimensiones para comprobar q los datos se cargaron correctamente
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# validacion cruzada
# dividimos el conjunto de entrenamiento en 5 partes manteniendo aproximadamente la misma proporcion
# de clases en cada particion
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

# guardamos las particiones q vamos a utilizar durante el calculo del fitness
folds = list(cv.split(X_train, y_train))

# inicializacion de la poblacion
def inicializar_poblacion(tam_poblacion,n_caracteristicas,prob_inicial):
    # cada individuo tiene un bit por cada caracteristica 0 significa q la caracteristica no se selecciona
    # 1 significa q la caracteristica si se selecciona
    poblacion = (np.random.rand(tam_poblacion,n_caracteristicas) < prob_inicial).astype(int)
    return poblacion

# decodificacion
def decodificar(individuo):

    # buscamos las posiciones donde el individuo tiene un 1
    # esas posiciones representan las caracteristicas seleccionadas
    indices = np.flatnonzero(individuo)
    return indices



# funcion de fitness
def calcular_fitness(individuo,X,y,folds):
    # decodificamos el individuo
    # obtenemos las posiciones de las caracteristicas q fueron seleccionadas
    indices = decodificar(individuo)
    # si no seleccionamos ninguna caracteristica, consideramos q no es una solucion valida
    if len(indices) == 0:
        return 0
    # seleccionamos las caracteristicas
    # nos quedamos solamente con las columnas seleccionadas por el individuo
    X_seleccionado = X[:, indices]
    # validacion cruzada
    accuracies = []
    # recorremos los 5 folds
    for indices_train, indices_val in folds:
        # separamos los datos de entrenamiento y validacion correspondientes al fold
        Xtr = X_seleccionado[indices_train]
        Xval = X_seleccionado[indices_val]
        ytr = y[indices_train]
        yval = y[indices_val]
        # arbol de decision
        # creamos un arbol para evaluar las caracteristicas seleccionadas
        modelo = DecisionTreeClassifier(random_state=0)
        # entrenamos el modelo
        modelo.fit(Xtr,ytr)
        # calculamos el accuracy sobre el conjunto de validacion
        accuracy = modelo.score(Xval,yval)
        accuracies.append(accuracy)
    # accuracy promedio
    # calculamos el promedio de los 5 folds
    accuracy_promedio = np.mean(accuracies)
    # penalizacion por cantidad de caracteristicas
    # contamos cuantas caracteristicas fueron seleccionadas
    cantidad_caracteristicas = len(indices)
    # calculamos q proporcion de todas las caracteristicas estamos utilizando
    proporcion = (cantidad_caracteristicas/ X.shape[1])
    # alpha determina cuanto penalizamos la utilizacion de muchas caracteristicas
    alpha = 0.1
    # fitness multiobjetivo
    # buscamos tener un accuracy alto utilizando la menor cantidad de caracteristicas posible
    fitness = (accuracy_promedio- alpha * proporcion)
    return fitness



# seleccion por competencias
def seleccion_competencias(poblacion,fitness,n_progenitores,k):
    progenitores = []
    for _ in range(n_progenitores):
        # elegimos k individuos al azar para q compitan entre ellos
        participantes = np.random.choice(len(poblacion),size=k,replace=False)
        # gana el individuo q tenga mayor fitness
        ganador = participantes[np.argmax(fitness[participantes])]
        # guardamos una copia del ganador como progenitor
        progenitores.append(poblacion[ganador].copy())
    return np.array(progenitores)



# cruza simple
def cruza_simple(padre1,padre2):
    # cantidad de bits del cromosoma
    n_bits = len(padre1)
    # elegimos aleatoriamenteel punto donde vamos a cortar
    punto = np.random.randint(1,n_bits)
    # cada hijo recibe una parte de cada padre
    hijo1 = np.concatenate([ padre1[:punto],padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto],padre1[punto:]])
    return hijo1, hijo2


# mutacion
def mutacion(individuo,prob_mutacion):
    # hacemos una copia para no modificar directamente el individuo original
    individuo = individuo.copy()
    # recorremos todos los bits del individuo
    for i in range(len(individuo)):
        # con cierta probabilidad invertimos el bit
        if np.random.rand() < prob_mutacion:
            individuo[i] = (1 - individuo[i])
    return individuo


# parametros


# fijamos la semilla para poder repetir exactamente el mismo experimento
np.random.seed(42)

# obtenemos la cantidad de caracteristicas a partir de los datos de entrenamiento
n_caracteristicas = X_train.shape[1]

# cantidad de individuos de la poblacion
tam_poblacion = 40

# cantidad de progenitores q vamos a seleccionar
n_progenitores = 10

# cantidad de individuos q participan en cada competencia
k = 3


# probabilidad inicial
# al comenzar queremos q los individuos seleccionen pocas caracteristicas
# como hay muchas caracteristicas, usamos una probabilidad inicial baja

prob_inicial = 0.01

# probabilidad de mutacion
# buscamos q en promedio se produzca aproximadamente una mutacion por individuo
prob_mutacion = (1 / n_caracteristicas)

# cantidad maxima de generaciones
max_generaciones = 30



# poblacion inicial
# generamos la primera poblacion utilizando la probabilidad inicial definida
poblacion = inicializar_poblacion(tam_poblacion,n_caracteristicas,prob_inicial)


# historial
# guardamos la evolucion del mejor fitness
historial_fitness = []
# guardamos la cantidad de caracteristicas seleccionadas por el mejor individuo
historial_caracteristicas = []
# mejor solucion encontrada hasta el momento
mejor_individuo = None
# comenzamos con un fitness muy bajo para q cualquier solucion valida pueda superarlo
mejor_fitness_global = -np.inf

# ciclo principal
for generacion in range(max_generaciones):
    # evaluacion
    # calculamos el fitness de todos los individuos de la poblacion
    fitness = np.array([calcular_fitness(individuo,X_train,y_train,folds)for individuo in poblacion])

    # mejor individuo de la generacion buscamos la posicion del individuo con mayor fitness
    indice_mejor = np.argmax(fitness)
    # obtenemos su fitness
    mejor_fitness = fitness[indice_mejor]

    # guardamos una copia del individuo
    mejor_actual = poblacion[indice_mejor].copy()

    # contamos cuantas caracteristicas selecciono
    cantidad_caracteristicas = np.sum(mejor_actual)

    # guardamos el historial
    historial_fitness.append(mejor_fitness)
    historial_caracteristicas.append(cantidad_caracteristicas)

    # guardamos el mejor global
    # si el mejor de esta generacion supera al mejor encontrado hasta ahora, lo guardamos
    if mejor_fitness > mejor_fitness_global:
        mejor_fitness_global = (mejor_fitness)
        mejor_individuo = (mejor_actual.copy())

    # mostramos informacion
    print(
        f"generacion {generacion}: "
        f"fitness = {mejor_fitness:.5f}, "
        f"caracteristicas = "
        f"{cantidad_caracteristicas}"
    )

    # criterio de parada
    # si encontramos un fitness suficientemente alto, terminamos el algoritmo
    if mejor_fitness >= 0.95:
        print("\ncriterio de parada alcanzado.")
        break

    # elitismo
    # guardamos el mejor individuo para asegurarnos de q no se pierda en la siguiente generacion
    elitista = poblacion[indice_mejor].copy()
    # sacamos el elitista
    # lo quitamos temporalmente antes de realizar la seleccion de progenitores
    poblacion_sin_elitista = np.delete(poblacion,indice_mejor,axis=0)
    fitness_sin_elitista = np.delete(fitness,indice_mejor)



    # seleccion de progenitores
    # seleccionamos los progenitores mediante competencias

    progenitores = (seleccion_competencias(poblacion_sin_elitista,fitness_sin_elitista,n_progenitores,k))

    # generacion de hijos
    hijos = []

    # necesitamos completar la poblacion manteniendo un individuo para el elitismo
    # por eso generamos:
    # 40 - 1 = 39 hijos
    # usamos while porque cada cruza genera dos hijos y 39 no es un numero par

    while len(hijos) < (tam_poblacion - 1):

        # elegimos dos progenitores de los 10 seleccionados
        indice_padre1 = np.random.randint(0,n_progenitores)
        indice_padre2 = np.random.randint(0,n_progenitores)

        padre1 = progenitores[indice_padre1]
        padre2 = progenitores[indice_padre2]


        # realizamos la cruza
        hijo1, hijo2 = cruza_simple(padre1,padre2)


        # aplicamos mutacion a los hijos
        hijo1 = mutacion(hijo1,prob_mutacion)
        hijo2 = mutacion(hijo2,prob_mutacion)

        # guardamos los hijos
        hijos.append(hijo1)
        hijos.append(hijo2)


    # convertimos la lista a un array
    # como while puede generar 40 hijos en lugar de 39, recortamos la lista para mantener exactamente 39

    hijos = np.array(hijos[:tam_poblacion - 1])
    # nueva poblacion


    # incorporamos el elitista y los hijos 1 elitista + 39 hijos = 40 individuos

    poblacion = np.vstack([elitista,hijos])


#resultado final
# obtenemos los indices de las caracteristicas seleccionadas por el mejor individuo global
indices_mejores = decodificar(mejor_individuo)

print()
print("==============================")
print("resultado final")
print("==============================")

# mostramos el mejor fitness encontrado
print(f"fitness: "f"{mejor_fitness_global:.5f}")

# mostramos la cantidad de caracteristicas seleccionadas
print(f"cantidad de caracteristicas: "f"{len(indices_mejores)}")

# mostramos cuales fueron las caracteristicas seleccionadas
print("caracteristicas seleccionadas:")
print(indices_mejores)

# evaluacion sobre el conjunto de test

# obtenemos nuevamente las posiciones de las caracteristicas seleccionadas por el mejor individuo encontrado
indices_mejores = decodificar(mejor_individuo)
# seleccionamos solamente esas caracteristicas
# usamos exactamente las mismas caracteristicas tanto en train como en test
X_train_seleccionado = X_train[:,indices_mejores]
X_test_seleccionado = X_test[:,indices_mejores]
# mostramos las dimensiones para comprobar q ahora trabajamos solamente con las 72 caracteristicas
print()
print("==============================")
print("datos seleccionados")
print("==============================")
print(f"X_train seleccionado: "f"{X_train_seleccionado.shape}")
print(f"X_test seleccionado: "f"{X_test_seleccionado.shape}")

#entrenar el modelo final
# creamos un nuevo arbol de decision
modelo_final = DecisionTreeClassifier(random_state=0)

# entrenamos el modelo utilizando todas las muestras de entrenamiento
# pero solamente las caracteristicas seleccionadas
modelo_final.fit(X_train_seleccionado,y_train)

#evaluar sobre test
# calculamos el accuracy sobre las 34 muestras q nunca fueron utilizadas durante la seleccion
# de caracteristicas
accuracy_test = modelo_final.score(X_test_seleccionado,y_test)

# resultado
print()
print("==============================")
print("resultado sobre test")
print("==============================")
print(f"caracteristicas utilizadas: "f"{len(indices_mejores)}")
print(f"accuracy sobre test: "f"{accuracy_test:.5f}")
print(f"porcentaje de aciertos: "f"{accuracy_test * 100:.2f}%")