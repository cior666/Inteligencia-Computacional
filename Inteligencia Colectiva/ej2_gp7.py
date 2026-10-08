import numpy as np
import pandas as pd


# ============================================================
# CARGAR LOS DATOS
# ============================================================

# ruta donde se encuentra el archivo gr17.csv
# el archivo contiene la matriz de distancias entre las ciudades
ruta = r"C:\Users\conra\OneDrive\Desktop\Facu Conrado\CUARTO AÑO\Inteligencia Computacional\Inteligencia Colectiva\gr17.csv"

# cargar la matriz de distancias
# cada posicion D[i,j] representa la distancia entre
# la ciudad i y la ciudad j
D = pd.read_csv(ruta, header=None).values

# cantidad de ciudades del problema
n = len(D)

print("Dimensiones de la matriz:", D.shape)


# ============================================================
# SISTEMA DE HORMIGAS
# ============================================================

def sistema_hormigas(
    D,
    n_hormigas=17,
    alpha=1,
    beta=1,
    rho=0.5,
    Q=1,
    max_iter=500
):

    # cantidad de ciudades
    n = len(D)

    # generador de numeros aleatorios
    # se fija la semilla para poder repetir el experimento
    # obteniendo siempre los mismos resultados
    rng = np.random.default_rng(0)


    # ========================================================
    # INICIALIZACION DE FEROMONAS
    # ========================================================

    # sigma[i,j] representa la cantidad de feromona
    # asociada al camino entre la ciudad i y la ciudad j
    #
    # inicialmente asignamos valores aleatorios entre 0 y 1
    sigma = rng.uniform(0, 1, (n, n))

    # no tiene sentido tener feromona desde una ciudad
    # hacia ella misma
    np.fill_diagonal(sigma, 0)


    # ========================================================
    # INFORMACION HEURISTICA
    # ========================================================

    # eta[i,j] representa la informacion heuristica
    # asociada al camino entre i y j
    #
    # en TSP utilizamos:
    #
    # eta[i,j] = 1 / distancia(i,j)
    #
    # por lo tanto, cuanto menor sea la distancia,
    # mayor sera la informacion heuristica
    eta = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            # evitamos dividir por cero
            if D[i, j] != 0:
                eta[i, j] = 1 / D[i, j]


    # ========================================================
    # MEJOR SOLUCION ENCONTRADA
    # ========================================================

    # al principio todavia no tenemos ningun camino
    mejor_camino = None

    # comenzamos con una longitud infinita
    # cualquier camino encontrado sera mejor que este
    mejor_longitud = np.inf


    # historial utilizado para analizar
    # la evolucion de la mejor solucion
    historial = []


    # ========================================================
    # ITERACIONES DEL ALGORITMO
    # ========================================================

    for t in range(max_iter):

        # guardamos los caminos construidos por
        # todas las hormigas en esta iteracion
        caminos = []

        # guardamos la longitud de cada camino
        longitudes = []


        # ====================================================
        # CADA HORMIGA CONSTRUYE UN CAMINO
        # ====================================================

        for k in range(n_hormigas):

            # todas las hormigas comienzan en la ciudad 0
            camino = [0]

            # conjunto de ciudades que todavia
            # no fueron visitadas
            no_visitadas = set(range(1, n))


            # ------------------------------------------------
            # CONSTRUCCION DEL CAMINO
            # ------------------------------------------------

            # mientras existan ciudades sin visitar
            while no_visitadas:

                # ciudad en la que se encuentra actualmente
                i = camino[-1]

                # convertimos las ciudades disponibles
                # a un array para poder trabajar con numpy
                candidatos = np.array(list(no_visitadas))


                # ------------------------------------------------
                # CALCULO DE LA PROBABILIDAD DE TRANSICION
                # ------------------------------------------------

                # para cada ciudad candidata calculamos:
                #
                # feromona^alpha * heuristica^beta
                #
                # sigma representa la experiencia acumulada
                # por las hormigas
                #
                # eta representa la conveniencia heuristica
                # de elegir ese camino
                numerador = (
                    sigma[i, candidatos] ** alpha
                    * eta[i, candidatos] ** beta
                )


                # normalizamos para obtener probabilidades
                # cuya suma sea igual a 1
                probabilidades = (
                    numerador / np.sum(numerador)
                )


                # elegimos aleatoriamente la siguiente ciudad
                # utilizando las probabilidades calculadas
                j = rng.choice(
                    candidatos,
                    p=probabilidades
                )


                # agregamos la ciudad elegida al camino
                camino.append(j)

                # eliminamos la ciudad de las no visitadas
                no_visitadas.remove(j)


            # ------------------------------------------------
            # VOLVER A LA CIUDAD INICIAL
            # ------------------------------------------------

            # el TSP requiere formar un ciclo
            # por eso la hormiga debe volver a la ciudad 0
            camino.append(0)


            # =================================================
            # CALCULAR LONGITUD DEL CAMINO
            # =================================================

            # acumulador de la distancia total
            longitud = 0

            # recorremos todos los saltos realizados
            for q in range(len(camino) - 1):

                # ciudad de origen
                i = camino[q]

                # ciudad de destino
                j = camino[q + 1]

                # sumamos la distancia entre ambas ciudades
                longitud += D[i, j]


            # guardamos el camino y su longitud
            caminos.append(camino)
            longitudes.append(longitud)


            # =================================================
            # ACTUALIZAR MEJOR SOLUCION GLOBAL
            # =================================================

            # si el camino actual es mejor que el mejor
            # encontrado hasta el momento, lo guardamos
            if longitud < mejor_longitud:

                mejor_longitud = longitud

                # usamos copy para guardar una copia
                # independiente del camino
                mejor_camino = camino.copy()


        # ====================================================
        # EVAPORACION DE FEROMONAS
        # ====================================================

        # una parte de la feromona desaparece en cada iteracion
        #
        # sigma = (1-rho) * sigma
        #
        # rho controla la tasa de evaporacion
        #
        # un rho grande -> mayor evaporacion
        # un rho pequeño -> menor evaporacion
        sigma *= (1 - rho)


        # ====================================================
        # DEPOSITO GLOBAL DE FEROMONAS
        # ====================================================

        # cada hormiga deposita feromona en los caminos
        # que utilizo
        #
        # la cantidad depositada es:
        #
        # deposito = Q / longitud
        #
        # por lo tanto, los caminos mas cortos
        # reciben mayor cantidad de feromona
        for camino, longitud in zip(caminos, longitudes):

            deposito = Q / longitud


            # recorremos todos los enlaces del camino
            for q in range(len(camino) - 1):

                # ciudad de origen
                i = camino[q]

                # ciudad de destino
                j = camino[q + 1]


                # depositamos feromona en ambos sentidos
                sigma[i, j] += deposito
                sigma[j, i] += deposito


        # ====================================================
        # GUARDAR EVOLUCION
        # ====================================================

        # guardamos la mejor longitud encontrada
        # hasta esta iteracion
        #
        # esto permite analizar la convergencia
        historial.append(mejor_longitud)


    # devolvemos:
    #
    # mejor camino encontrado
    # mejor longitud encontrada
    # historial de convergencia
    return mejor_camino, mejor_longitud, historial


# ============================================================
# EJECUTAR EL ALGORITMO
# ============================================================

camino, longitud, historial = sistema_hormigas(D)


# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\nMejor camino encontrado:")

# convertimos los valores a int para mostrar
# las ciudades de forma mas clara
print([int(ciudad) for ciudad in camino])


print("\nLongitud:")

print(longitud)


print("\nHistorial de evolucion:")

# mostramos la evolucion como enteros
print([int(i) for i in historial])