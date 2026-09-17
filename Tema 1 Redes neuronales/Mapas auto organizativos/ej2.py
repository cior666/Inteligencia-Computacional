import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# cargar los datos
# --------------------------------------------------

# cargamos el conjunto de entrenamiento y prueba
trn = np.loadtxt(
    r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Tema 1 Redes neuronales\Mapas auto organizativos\iris81_trn.csv",
    delimiter=","
)

tst = np.loadtxt(
    r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Tema 1 Redes neuronales\Mapas auto organizativos\iris81_tst.csv",
    delimiter=","
)


# --------------------------------------------------
# separar datos y clases
# --------------------------------------------------

# las primeras 4 columnas son las caracteristicas de iris
datos = trn[:, :4]

# las ultimas 3 columnas indican la clase
clases_codigo = trn[:, 4:]


# convertimos el codigo de tres valores a un numero de clase
# 0 = setosa
# 1 = versicolor
# 2 = virginica
clases = np.full(len(clases_codigo), -1)
clases[np.all(clases_codigo == [-1, -1, 1], axis=1)] = 0
clases[np.all(clases_codigo == [-1, 1, -1], axis=1)] = 1
clases[np.all(clases_codigo == [1, -1, -1], axis=1)] = 2
nombres_clases = ["setosa", "versicolor", "virginica"]

# neurona ganadora
def neurona_ganadora(x, pesos):
    # calculamos la distancia entre el patron y todas las neuronas
    distancias = np.linalg.norm(pesos - x, axis=1)
    # la neurona ganadora es la q tiene menor distancia
    return np.argmin(distancias)


# entrenar som 2d

def entrenar_som_2d(datos, filas, columnas):
    cantidad_neuronas = filas * columnas
    # inicializamos los pesos aleatoriamente
    # cada neurona tiene 4 pesos porque iris tiene 4 caracteristicas
    pesos = np.random.uniform(-0.5,0.5,size=(cantidad_neuronas, datos.shape[1]))

    # guardamos la posicion de cada neurona dentro del mapa
    posiciones = np.array([[i, j]
                           for i in range(filas)
                           for j in range(columnas)])
    
    # etapa 1: ordenamiento global
    epocas_1 = 100
    # usamos una vecindad grande al principio
    radio_1 = max(filas, columnas) / 2
    # tasa de aprendizaje inicial
    tasa1 = 0.8

    for epoca in range(epocas_1):
        # mezclamos los patrones para cambiar el orden de entrenamiento
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            # buscamos la neurona ganadora
            ganadora = neurona_ganadora(x, pesos)
            # obtenemos la posicion de la ganadora en el mapa
            posicion_ganadora = posiciones[ganadora]
            # recorremos todas las neuronas
            for j in range(cantidad_neuronas):
                # calculamos la distancia entre neuronas dentro del mapa
                distancia = np.linalg.norm(posiciones[j] - posicion_ganadora)
                # si esta dentro de la vecindad, actualizamos su peso
                if distancia <= radio_1:
                    pesos[j] += tasa1 * (x - pesos[j])


    # etapa 2: transicion

    epocas_2 = 100
    radio_inicial = radio_1
    radio_final = 1
    tasa_inicial = tasa1
    tasa_final = 0.1

    for epoca in range(epocas_2):

        # reducimos el radio de forma lineal
        radio = (radio_inicial+ (radio_final - radio_inicial)* epoca / (epocas_2 - 1))

        # reducimos la tasa de aprendizaje de forma lineal
        tasa = (tasa_inicial+(tasa_final - tasa_inicial)* epoca / (epocas_2 - 1))
        indices = np.random.permutation(len(datos))

        for indice in indices:
            x = datos[indice]
            # buscamos la neurona ganadora
            ganadora = neurona_ganadora(x, pesos)
            posicion_ganadora = posiciones[ganadora]
            for j in range(cantidad_neuronas):
                # distancia dentro del mapa
                distancia = np.linalg.norm(posiciones[j] - posicion_ganadora)
                # actualizamos las neuronas q siguen dentro de la vecindad
                if distancia <= radio:
                    pesos[j] += tasa * (x - pesos[j])


    # etapa 3: ajuste fino
    epocas_3 = 300
    # con radio 0 solo se actualiza la neurona ganadora
    tasa = 0.05
    for epoca in range(epocas_3):
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            # buscamos la neurona ganadora
            ganadora = neurona_ganadora(x, pesos)
            # actualizamos solamente la neurona ganadora
            pesos[ganadora] += tasa * (x - pesos[ganadora])
    return pesos


# k-medias

def k_medias(datos, k, max_epocas=100):
    # elegimos k patrones al azar para usar como centroides iniciales
    indices = np.random.choice(len(datos),k,replace=False)

    centroides = datos[indices].copy()
    for epoca in range(max_epocas):
        # calculamos la distancia de cada patron a cada centroide
        distancias = np.linalg.norm(datos[:, np.newaxis, :] - centroides[np.newaxis, :, :],axis=2)

        # cada patron se asigna al centroide mas cercano
        grupos = np.argmin(distancias, axis=1)
        # guardamos los centroides anteriores
        centroides_anterior = centroides.copy()
        # actualizamos cada centroide
        for j in range(k):

            patrones_grupo = datos[grupos == j]
            # si el grupo tiene patrones, calculamos su promedio
            if len(patrones_grupo) > 0:
                centroides[j] = np.mean(
                    patrones_grupo,
                    axis=0
                )
        # si los centroides dejaron de cambiar, terminamos
        if np.allclose(
            centroides,
            centroides_anterior
        ):
            break

    return centroides, grupos


# entrenar modelos
print("entrenando k-medias...")
# iris tiene 3 clases de referencia
k = 3

centroides, grupos_kmeans = k_medias(datos,k)
print("entrenando som...")
# usamos 10x10 = 100 neuronas
filas = 10
columnas = 10
pesos_som = entrenar_som_2d(datos,filas,columnas)
# buscamos la neurona ganadora para cada patron
ganadoras_som = np.array([neurona_ganadora(x, pesos_som)
    for x in datos
])

# matriz de contingencia k-medias vs clases
def matriz_contingencia(grupos, clases, cantidad_grupos, cantidad_clases):
    matriz = np.zeros((cantidad_grupos, cantidad_clases),dtype=int)

    for grupo, clase in zip(grupos, clases):
        matriz[grupo, clase] += 1
    return matriz


matriz_kmeans_clases = matriz_contingencia(grupos_kmeans,clases,3,3)

print("\nmatriz de contingencia: k-medias vs clases")
print(matriz_kmeans_clases)

# matriz de contingencia som vs clases
matriz_som_clases = matriz_contingencia(ganadoras_som,clases,filas * columnas,3)
print("\nmatriz de contingencia: som vs clases")
print(matriz_som_clases)


# matriz de contingencia k-medias vs som

# para comparar k-medias con som contamos cuantos patrones de cada grupo de k-medias
# terminaron en cada neurona del som

#La matriz de contingencia entre K-medias y SOM muestra cómo se distribuyen los patrones de cada grupo de K-medias entre las 100 neuronas del SOM. 
#Cada fila corresponde a un grupo de K-medias y cada columna a una neurona del SOM
matriz_kmeans_som = matriz_contingencia(grupos_kmeans,ganadoras_som,3,filas * columnas)
print("\nmatriz de contingencia: k-medias vs som")
print(matriz_kmeans_som)
#Cada numero indica cuantos patrones que pertenecen a un grupo de K-medias terminaron teniendo como neurona ganadora a una determinada neurona del SOM.
#por ejemplo para la primera fila: [[0 0 0 0 0 3 0 0 0 0 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 6 0 0 0 0 0 0 0 0 2 0 es M[0,5]=3
#3 patrones que K-medias coloco en el grupo 0 fueron representados por la neurona 5 del SOM

# frecuencia de activacion de cada neurona
# contamos cuantas veces gano cada neurona
frecuencias = np.bincount(ganadoras_som,minlength=filas * columnas)

print("\nfrecuencia de activacion de las neuronas:")
print(frecuencias)
# clase de cada neurona

clase_neurona = np.full(filas * columnas,-1)
for j in range(filas * columnas):
    # buscamos las clases de los patrones q activaron esta neurona
    clases_neurona = clases[ganadoras_som == j]
    if len(clases_neurona) > 0:
        # asignamos la clase q aparece mas veces
        clase_neurona[j] = np.bincount(clases_neurona,minlength=3).argmax()


# graficos de k-medias y som
# elegimos dos dimensiones para visualizar usamos longitud del sepalo y longitud del petalo
dim1 = 0
dim2 = 2

# grafico de k-medias
plt.figure(figsize=(7, 7))
plt.scatter(
    datos[:, dim1],
    datos[:, dim2],
    c=grupos_kmeans,
    cmap="tab10",
    s=20
)

plt.scatter(
    centroides[:, dim1],
    centroides[:, dim2],
    marker="x",
    color="black",
    s=100
)

plt.xlabel("longitud del sepalo")
plt.ylabel("longitud del petalo")
plt.title("k-medias")
plt.grid()
plt.axis("equal")


# grafico del som
plt.figure(figsize=(7, 7))
plt.scatter(
    datos[:, dim1],
    datos[:, dim2],
    c=ganadoras_som,
    cmap="tab20",
    s=20
)

# mostramos los pesos de las neuronas usando las dos dimensiones seleccionadas
pesos_2d = pesos_som[:, [dim1, dim2]]
plt.scatter(
    pesos_2d[:, 0],
    pesos_2d[:, 1],
    marker="x",
    color="black",
    s=50
)

plt.xlabel("longitud del sepalo")
plt.ylabel("longitud del petalo")
plt.title("som")
plt.grid()
plt.axis("equal")


# grafico de frecuencia de activacion
pesos_mapa = pesos_2d.reshape(filas,columnas,2)

frecuencias_mapa = frecuencias.reshape(filas,columnas)
clases_mapa = clase_neurona.reshape(filas,columnas)

plt.figure(figsize=(9, 9))
# cada neurona tiene un color segun su frecuencia
plt.scatter(
    pesos_mapa[:, :, 0],
    pesos_mapa[:, :, 1],
    c=frecuencias_mapa,
    cmap="viridis",
    s=250
)

# conectamos las neuronas vecinas horizontalmente
for i in range(filas):
    for j in range(columnas - 1):
        p1 = pesos_mapa[i, j]
        p2 = pesos_mapa[i, j + 1]
        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "k-",
            linewidth=0.5
        )


# conectamos las neuronas vecinas verticalmente
for i in range(filas - 1):
    for j in range(columnas):
        p1 = pesos_mapa[i, j]
        p2 = pesos_mapa[i + 1, j]
        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "k-",
            linewidth=0.5
        )

# mostramos la clase de cada neurona
for i in range(filas):
    for j in range(columnas):
        clase = clases_mapa[i, j]
        if clase != -1:
            plt.text(
                pesos_mapa[i, j, 0],
                pesos_mapa[i, j, 1],
                nombres_clases[clase],
                fontsize=6,
                ha="center",
                va="center"
            )


plt.xlabel("longitud del sepalo")
plt.ylabel("longitud del petalo")
plt.title("som - frecuencia de activacion y clase")
plt.colorbar(label="frecuencia")
plt.grid()
plt.axis("equal")
plt.show()

#viendo las matrices, por ejemplo en el caso de la k-media vs clases reales
#obtenemos una matriz: 
#[[ 0  0 24]
# [45  0  0]
# [ 0 32 10]]
#sabesmos q la columna 0 corresponde a setosa, la 1 a versicolor la 2 a virginica
#por lo q interpretando, obtenemos que: el grupo 0 tiene 24 patrones virginica
#el grupo 1 contiene 45 patrones de setosa y el grupo 2 contiene 32 versicolor y 10 virginica
#esto tb lo vemos en el grafico donde setosa queda correctamente separada
#pero versicolor y virginica se superponen

#para ver la suma de clasificaciones del som:
# comprobacion de las clases
print("\ncomprobacion de clases:")
for i in range(3):
    cantidad = np.sum(clases == i)
    cantidad_som = np.sum(matriz_som_clases[:, i])
    print(nombres_clases[i],"-> datos:",cantidad,"som:",cantidad_som)
    if cantidad == cantidad_som:
        print("  coincide")
    else:
        print("  no coincide")

#viendo los prints comprobamos q recuperamos las cantidades de patrones de cada clase
#q aparecen en el conjunto q usamos.

#La frecuencia de activación responde a: "¿Cuántos patrones fueron representados por esta neurona?"
#la clase de la neurona responde a: "¿Qué clase aparece con mayor frecuencia entre los patrones que activaron esta neurona?"
print("\nfrecuencia de activacion de las neuronas:")
for i in range(len(frecuencias)):
    print("neurona", i, "->", frecuencias[i], "patrones")

print("\ntotal de patrones:", len(datos))
print("suma de frecuencias:", np.sum(frecuencias))
if np.sum(frecuencias) == len(datos):
    print("la suma coincide con el total de patrones")
else:
    print("la suma no coincide")
