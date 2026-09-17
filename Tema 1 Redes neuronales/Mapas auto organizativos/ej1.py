import numpy as np
import matplotlib.pyplot as plt

def neurona_ganadora(x, pesos):

    # distancia euclídea entre x y cada neurona
    distancias = np.linalg.norm(pesos - x, axis=1)
    # devuelve la posicion de la neurona ganadora esto esta igual q en el apunte
    return np.argmin(distancias)

def graficar_som_2d(datos, pesos, filas, columnas, etapa):
    plt.figure(figsize=(7, 7))
    # calculamos la ganadora para cada punto de datos
    ganadoras = []
    for x in datos:
        ganadora = neurona_ganadora(x, pesos)
        ganadoras.append(ganadora)
    ganadoras = np.array(ganadoras)
    # el color de cada punto de datos depende de la neurona ganadora
    plt.scatter(datos[:, 0],datos[:, 1],c=ganadoras,cmap="tab20",s=10)
    plt.scatter(pesos[:, 0],pesos[:, 1],marker="x",color="black",s=50)
    # pasamos los pesos de 100x2 a una matriz de 10x10x2
    # esto nos permite trabajar con la posicion de cada neurona en el mapa
    pesos_mapa = pesos.reshape(filas, columnas, 2)
   # uniones horizontales
    for i in range(filas):
        for j in range(columnas - 1):
            p1 = pesos_mapa[i, j]
            p2 = pesos_mapa[i, j + 1]
            plt.plot([p1[0], p2[0]],[p1[1], p2[1]],"k-",linewidth=0.8)
    # uniones verticales
    for i in range(filas - 1):
        for j in range(columnas):
            p1 = pesos_mapa[i, j]
            p2 = pesos_mapa[i + 1, j]
            plt.plot([p1[0], p2[0]],[p1[1], p2[1]],"k-",linewidth=0.8)

    plt.title("SOM 2D - Etapa " + str(etapa))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()

# ENTRENAR SOM 2D
def entrenar_som_2d(datos, filas, columnas):
    cantidad_neuronas = filas * columnas
    pesos = np.random.uniform(-0.5,0.5,size=(cantidad_neuronas, 2))
    # guardamos la posicion de cada neurona dentro del mapa
    # esto se usa para calcular q neuronas son vecinas
    posiciones = np.array([[i, j]
        for i in range(filas)
        for j in range(columnas)
    ])
   # ordenamiento global
    epocas_1 = 100
    # usamos una vecindad grande para ordenar todo el mapa
    radio_1 = cantidad_neuronas / 2
    # tasa de aprendizaje inicial
    tasa1 = 0.8
    for epoca in range(epocas_1):
        # mezclo datos para que no se entrenen siempre en el mismo orden
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            # buscamos la neurona mas cercana al patron
            ganadora = neurona_ganadora(x, pesos)
            # obtenemos la posicion de la neurona ganadora en el mapa
            posicion_ganadora = posiciones[ganadora]
            # recorremos todas las neuronas para buscar las vecinas
            for j in range(cantidad_neuronas):
            # calculamos la distancia entre las posiciones del mapa
                distancia = np.linalg.norm(posiciones[j] - posicion_ganadora)
            # si esta dentro del radio, tambien se actualiza
                if distancia <= radio_1:
                    pesos[j] += tasa1 * (x - pesos[j])
    # mostramos como quedo el mapa despues del ordenamiento global
    graficar_som_2d(datos,pesos,filas,columnas,1)

    # transicion
    epocas_2 = 100
# el radio va disminuyendo desde el valor inicial hasta 1
    radio_inicial = radio_1
    radio_final = 1
# la tasa tambien va disminuyendo
    tasa_inicial = tasa1
    tasa_final = 0.1
    for epoca in range(epocas_2):
        # hacemos una interpolacion lineal para reducir el radio
        radio = radio_inicial + (radio_final - radio_inicial) * epoca / (epocas_2 - 1)
        # hacemos lo mismo con la tasa de aprendizaje
        tasa = tasa_inicial + (tasa_final - tasa_inicial) * epoca / (epocas_2 - 1)
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            #buscamos la neurona ganadora
            ganadora = neurona_ganadora(x, pesos)
            posicion_ganadora = posiciones[ganadora]
            for j in range(cantidad_neuronas):
                #calculamos la distancia dentro del mapa
                distancia = np.linalg.norm(posiciones[j] - posicion_ganadora)
                #actualizamos las neuronas dentro de la vecindad
                if distancia <= radio:
                    pesos[j] += tasa * (x - pesos[j])
    graficar_som_2d(datos,pesos,filas,columnas,2)

    # ajuste fino
    epocas_3 = 300
    # radio 0 significa q solo se actualiza la neurona ganadora
    radio = 0   # solo gana la ganadora
    tasa = 0.05 #usamos una tasa de aprendizaje mas peque
    for epoca in range(epocas_3):
        indices = np.random.permutation(len(datos))
        for indice in indices:
            x = datos[indice]
            ganadora = neurona_ganadora(x, pesos)
            pesos[ganadora] += tasa * (x - pesos[ganadora])
    graficar_som_2d(datos,pesos,filas,columnas,3)
    return pesos



circulos = np.loadtxt("circulo.csv",delimiter=",")
T = np.loadtxt("te.csv",delimiter=",")
filas = 10
columnas = 10
print("Entrenando SOM con círculos")
pesos_circulos = entrenar_som_2d(circulos,filas,columnas)
print("Entrenando SOM con T")
pesos_T = entrenar_som_2d(T,filas,columnas)
plt.show()