import numpy as np
# funcion objetivo del ejercicio 1 ii de la GTP6
# el objetivo es encontrar el valor de x e y que minimiza f(x,y)
def funcion(x, y):
    # calculamos x^2 + y^2
    r = x**2 + y**2
    # devolvemos el valor de la funcion
    return r**0.25 * (np.sin(50 * r**0.1)**2 + 1)

# parametros del problema
# limites del espacio de busqueda
xmin = -100
xmax = 100
ymin = -100
ymax = 100

# cantidad de particulas del enjambre
N = 30

# constantes de aceleracion
# c1 controla la influencia de la experiencia personal
# c2 controla la influencia de la experiencia social
c1 = 2
c2 = 2

# cantidad maxima de iteraciones del algoritmo
max_iter = 1000

# velocidad de convergencia
# valor de referencia para medir la velocidad de convergencia usamos el mismo umbral q en el algoritmo genetico
# para poder comparar cuantos pasos necesita cada metodo
umbral_convergencia = 0.1

# guardamos la primera iteracion en la q se alcanza el umbral
iteracion_convergencia = None

# inicializacion
# generador de numeros aleatorios se fija la semilla para poder repetir el experimento obteniendo siempre la misma secuencia de numeros aleatorios
rng = np.random.default_rng(0)

# posiciones iniciales de las particulas
# cada particula tiene una posicion x y una posicion y
# ambas se generan aleatoriamente dentro del dominio
x = rng.uniform(xmin, xmax, N)
y = rng.uniform(ymin, ymax, N)

# velocidades iniciales de las particulas
# todas comienzan con velocidad cero
vx = np.zeros(N)
vy = np.zeros(N)

# mejor historico de cada particula
# mejor posicion x personal de cada particula inicialmente es la propia posicion inicial
x_personal = x.copy()

# mejor posicion y personal de cada particula inicialmente es la propia posicion inicial
y_personal = y.copy()

# valor de la funcion en la mejor posicion personal f_personal[k] representa el mejor valor encontrado
# por la particula k
f_personal = funcion(x_personal, y_personal)

# mejor posicion global
# busco el indice de la particula que tiene el menor valor de la funcion
indice_mejor = np.argmin(f_personal)

# mejor posicion x encontrada por todo el enjambre
x_glob = x_personal[indice_mejor]

# mejor posicion y encontrada por todo el enjambre
y_glob = y_personal[indice_mejor]

# valor de la funcion en la mejor posicion global
f_y_glob = f_personal[indice_mejor]

# evolucion
# lista para guardar el mejor valor global encontrado en cada iteracion permite analizar la convergencia del algoritmo
historial = []

for t in range(max_iter):
    # actualizar mejores historicos
    # evaluo la funcion en la posicion actual de cada particula
    f_actual = funcion(x, y)

    # comparo la posicion actual de cada particula con su mejor posicion personal encontrada hasta ahora
    # como estamos minimizando, una posicion es mejor cuando produce un valor menor de la funcion
    mejores = f_actual < f_personal

    # para las particulas que mejoraron, actualizo su mejor posicion personal
    x_personal[mejores] = x[mejores]
    y_personal[mejores] = y[mejores]

    # actualizo tambien el valor de la funcion asociado a su mejor posicion personal
    f_personal[mejores] = f_actual[mejores]

    # actualizar mejor global
    # busco cual es actualmente la mejor posicion personal de todo el enjambre
    indice_mejor = np.argmin(f_personal)

    # si la mejor posicion actual es mejor que la mejor posicion global que tenia guardada, actualizo la mejor posicion global
    if f_personal[indice_mejor] < f_y_glob:
        x_glob = x_personal[indice_mejor]
        y_glob = y_personal[indice_mejor]
        f_y_glob = f_personal[indice_mejor]

    # guardo el mejor valor global encontrado en esta iteracion para analizar la evolucion
    historial.append(f_y_glob)

    # velocidad de convergencia
    # verificamos si el mejor valor encontrado ya alcanzo el mismo nivel de calidad usado en el algoritmo genetico
    if (iteracion_convergencia is None and f_y_glob <= umbral_convergencia):
        # guardamos la cantidad de iteraciones necesarias
        iteracion_convergencia = t + 1

    # generar numeros aleatorios

    # genero un numero aleatorio entre 0 y 1 para cada particula
    # r1 controla de forma aleatoria la influencia de la experiencia personal
    r1 = rng.uniform(0, 1, N)

    # r2 controla de forma aleatoria la influencia
    # de la experiencia social
    r2 = rng.uniform(0, 1, N)

    # actualizar velocidad
    # ecuacion de actualizacion de la velocidad:
    # v(t+1) = v(t)+ c1*r1*(y - x)+ c2*r2*(y_glob - x)
    # como ahora tenemos dos dimensiones,
    # hacemos la actualizacion por separado
    # para x y para y

    # actualizacion de la velocidad en x
    vx = (vx+ c1 * r1 * (x_personal - x)+ c2 * r2 * (x_glob - x))
    # actualizacion de la velocidad en y
    vy = (vy+ c1 * r1 * (y_personal - y)+ c2 * r2 * (y_glob - y))

    # actualizar posicion
    # actualizo la posicion x utilizando la nueva velocidad
    x = x + vx
    # actualizo la posicion y utilizando la nueva velocidad
    y = y + vy

    # mantener las particulas dentro del dominio
    # las particulas deben permanecer dentro del intervalo definido para cada dimension
    # si una particula sale del intervalo, se fuerza su posicion al limite correspondiente
    x = np.clip(x, xmin, xmax)
    y = np.clip(y, ymin, ymax)

# resultado
# muestro la mejor posicion x encontrada
# por todo el enjambre
print("Mejor x encontrada:", x_glob)
# muestro la mejor posicion y encontrada por todo el enjambre
print("Mejor y encontrada:", y_glob)
# muestro el valor minimo de la funcion correspondiente a esa posicion
print("Valor de la funcion:", f_y_glob)

# velocidad de convergencia
print()
print("==============================")
print("velocidad de convergencia")
print("==============================")
print(f"umbral utilizado: f(x,y) <= {umbral_convergencia}")
print(f"iteraciones necesarias: {iteracion_convergencia}")

#el none que obtenemos en la cantidad de iteraaciones no significa que no funcione sino q el algoritmo no alcanzó 
#el umbral en ninguna de las mil iteraciones que realizo