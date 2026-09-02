import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ubi_train="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/Perceptrón multicapa/concent_trn.csv"
ubi_test="C:/Users/conra/OneDrive/Desktop/Facu Conrado/CUARTO AÑO/Inteligencia Computacional/Tema 1 Redes neuronales/Perceptrón multicapa/concent_tst.csv"

def sigmoide(v, b=1):
    return 2 / (1 + np.exp(-b * v)) - 1

# derivo la sigmoide y defino la funcion. dsp se va a usar para calcular el gradiente del error e ir actualizando los pesos
# si bien yo no busco q la derivada de la sigmoide sea minima, necesito saber su valor p ver hacia donde muevo los pesos
# para minimizar el error (eso le entendi al profe deberia preg)
def derivada_sigmoide(y):
    return 0.5 * (1 - y**2)

def signo(v):
    return np.where(v >= 0, 1, -1)

def leer_datos(ruta_archivo, n_salidas):
    matriz = np.loadtxt(ruta_archivo, delimiter=',')  
    n_columnas = matriz.shape[1]
    n_entradas = n_columnas - n_salidas
    x = matriz[:, 0:n_entradas]      # todas las filas, columnas de entrada
    y = matriz[:, n_entradas:n_columnas]   # todas las filas, columnas de salida
    return x, y

def inicializar_pesos(capas):
    W = []
    for p in range(len(capas) - 1): #recorro las conexiones entre las capas 
        #para saber cuantas neuronas tiene la q recibe info y cuantas neuronas tiene la q procesa y entrega a la siguiente capa defino:
        neuronas_entrada = capas[p] #por ej cuando p=0, neuronas de entradas son 2 (q es lo defini)
        neuronas_salida = capas[p + 1] #neuronas de salidas son 3 q son las q defini tb
        # +1 por el bias
        W.append(np.random.uniform(-0.5,0.5,(neuronas_salida, neuronas_entrada + 1)))
    return W

# PROPAGACION HACIA ADELANTE
def propagacion_adelante(x, W, activacion):
    # Potencial de activación
    v = W @ x #hacemos el prod matricial entre los pesos y las entradas, es lo q el profe habia explicado en el pizarron
    #para la salida hacemos como antes, pasamos el valor obtenido en la iteracion anterior por la funcion de activacion (sigmoide)
    y = activacion(v)
    return y

# DELTA DE LA CAPA DE SALIDA
def DeltaSalida(e, y_salida):
    return e * derivada_sigmoide(y_salida) #definida como en el apunte

# DELTA DE UNA CAPA OCULTA
#como hay q trabajar con el delta de la capa oculta, pero no tenemos directamente un error como en la salida.
#entonces, tomamos el delta de la capa siguiente y lo propagamos hacia atras usando los pesos que conectan ambas capas
# es el proceso q aparece en las diapos
def DeltaOculta(delta_siguiente, w_siguiente, y_capa): 
    #sacamos la columna del bias porq para propagar hacia neuronas anteriores no necesitamos uasrlo
    w_sin_bias = w_siguiente[:, 1:] 
    #propagamos el delta hacia atras, haciendo el prod matricial de nuevo, lo trasponemos para combinar bien los pesos con los deltas
    delta = w_sin_bias.T @ delta_siguiente #lo q sacamos aca es cuan responsable es una neurona oculta del error
    # multiplicamos por la derivada de la funcion de activacion, esto es por la formula q aparece en el apunte
    return delta * derivada_sigmoide(y_capa)

# RETROPROPAGACIÓN
def Retropropagacion(e, y, w):
    L = len(w) #matrices de pesos, para saber cuantos conjuntos de deltas calcular
    deltas = [None] * L #los inicializo como vacios
    # Delta de la capa de salida
    deltas[-1] = DeltaSalida(e,y[-1]) #busco el delta de la capa de salida (digamos la ultima)
    # busco los deltas para las capas q me faltan, como tengo la ultima primero hallo el de capa 2 y luego el de capa 1
    for p in range(L - 2, -1, -1):
        deltas[p] = DeltaOculta(deltas[p + 1],w[p + 1],y[p + 1])
    return deltas

# ACTUALIZACIÓN DE PESOS
def ActualizarPesos(w_capa,delta_capa,entrada_con_bias,vel_aprendizaje):
    return (w_capa+ vel_aprendizaje* np.outer(delta_capa, entrada_con_bias))
#esta es la formula del apunte:
# vel_aprendizaje → μ
# delta_capa → δj
# entrada_con_bias → yi
# w_capa → wji


# PROBAR LA RED CON LOS DATOS DE TEST
#este chequeo es casi lo mismo q los q hacia en la guia 1
def Probar(x_test, y_test, w, activacion):
    n_patrones = x_test.shape[0]
    aciertos = 0
    for n in range(n_patrones):
        # La entrada del patrón
        y = [x_test[n]]
        # Propagación hacia adelante
        for p in range(len(w)):
            # Agregamos bias
            x = np.concatenate(([-1], y[p]))
            # Salida de la capa
            y.append(propagacion_adelante(x,w[p],activacion))
        # Convertimos la salida continua a -1 o +1
        prediccion = signo(y[-1])

        # Comparamos con la salida deseada
        if np.array_equal(prediccion,y_test[n]):
            aciertos += 1
    return aciertos, n_patrones

def clasificar(x, w, activacion):
    y = [x]
    for p in range(len(w)):
        entrada_con_bias = np.concatenate(([-1], y[p]))
        y.append(propagacion_adelante(entrada_con_bias, w[p], activacion))
    return signo(y[-1])[0]

def graficar_resultado(w, activacion, x, d, x_min=0, x_max=1.2, y_min=0, y_max=1.2, resolucion=200):
    """Grafica el plano de decisión (fondo) y los datos reales (puntos) juntos"""
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucion),
                          np.linspace(y_min, y_max, resolucion))
    puntos = np.c_[xx.ravel(), yy.ravel()]
    clases = np.array([clasificar(p, w, activacion) for p in puntos]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, clases, levels=[-1.5, 0, 1.5], colors=['red', 'white'], alpha=0.4)


    clase_pos = d.flatten() == 1
    clase_neg = d.flatten() == -1
    plt.scatter(x[clase_pos, 0], x[clase_pos, 1], marker='x', color='black', label='+1')
    plt.scatter(x[clase_neg, 0], x[clase_neg, 1], marker='s', facecolors='none', edgecolors='red', label='-1')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.title('Plano de decisión + datos reales')
def graficar_activaciones(v_min=-5, v_max=5, n=500):
    v = np.linspace(v_min, v_max, n)

    plt.figure()
    plt.plot(v, sigmoide(v), label="Sigmoide bipolar")
    plt.plot(v, signo(v), label="Signo")
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.xlabel("v (potencial de activación)")
    plt.ylabel("φ(v)")
    plt.title("Funciones de activación")
    plt.legend()
    plt.grid(True)
    plt.show()

def graficar_datos(x, d):
    """Grafica patrones 2D coloreados por clase (d en {-1,1})"""
    plt.figure()
    clase_pos = d.flatten() == 1
    clase_neg = d.flatten() == -1

    plt.scatter(x[clase_pos, 0], x[clase_pos, 1], marker='x', color='black', label='+1')
    plt.scatter(x[clase_neg, 0], x[clase_neg, 1], marker='s', facecolors='none', edgecolors='red', label='-1')

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Distribución de clases')

def Ejercicio2(capas, ruta_train, ruta_test, vel_aprendizaje, max_epocas, err_umbral, activacion):
    #leemos los datos pasandole la ruta y la cantidad de salidas que tiene la red (capas[-1] es la cantidad de salidas)
    x_train, y_train = leer_datos(ruta_train, capas[-1])
    x_test, y_test = leer_datos(ruta_test, capas[-1])

    w = inicializar_pesos(capas) #inicializamos los pesos de la red (matriz de filas=neuronas de la capa siguiente, columnas=neuronas de la capa anterior + 1 para el sesgo)
    n_patrones = x_train.shape[0] #cantidad de patrones de entrenamiento

    for epoca in range(max_epocas):
        error_epoca = 0 #acumulador de error para esta época
        aciertos_epoca = 0 #acumulador de aciertos para esta época
        for n in range(n_patrones): #n recorre todos los patrones de entrenamiento
            #propagacion hacia adelante
            y=[x_train[n]] #y[0] es la entrada del patrón n
            for p in range(len(capas)-1): #p recorre todas las capas de la red
                #agregamos el sesgo a la entrada de la capa p
                x = np.concatenate(([-1], y[p])) #agregamos el sesgo x0=-1
                y.append(propagacion_adelante(x, w[p], activacion)) #y[p+1] es la salida de la capa p

            e = y_train[n] - y[-1]
            deltas = Retropropagacion(e, y, w) #calculamos los deltas de todas las capas

            #Actualizamos los pesos de todas las capas
            for p in range(len(w)): #p recorre todas las capas de la red
                entrada_con_bias = np.concatenate(([-1], y[p])) #agregamos el sesgo a la entrada de la capa p
                w[p] = ActualizarPesos(w[p], deltas[p], entrada_con_bias, vel_aprendizaje) #actualizamos los pesos de la capa p

            error_epoca += 0.5 * np.sum(e**2) #acumulamos el error cuadrático medio de todos los patrones de entrenamiento
            if np.array_equal(signo(y[-1]), y_train[n]):
                aciertos_epoca += 1

        tasa_aciertos = aciertos_epoca / n_patrones  
        print(f"Época {epoca}: xi = {error_epoca}, aciertos = {aciertos_epoca}, tasa = {tasa_aciertos:.2f}")

        if tasa_aciertos >= 0.95 or error_epoca < err_umbral:
            print(f"Convergencia alcanzada en la época {epoca}. Error: {error_epoca}")
            break
    else:
        print("No convergio en", max_epocas, "épocas. Error final:", error_epoca)

    aciertos, n_patrones = Probar(x_test, y_test, w, activacion)
    print(f"Aciertos: {aciertos}/{n_patrones} ({aciertos/n_patrones*100:.2f}%)")

    return w  # Devolvemos los pesos finales de la red

w = Ejercicio2([2,3,3,1], ubi_train, ubi_test, 0.01, 3000, 0.05, sigmoide)

print("Pesos finales:")
for i, wp in enumerate(w):
    print(f"  W[{i}] =\n{wp}")
x, d = leer_datos('concent_tst.csv', 1)
graficar_resultado(w, sigmoide, x, d)
plt.show()

#al ver la grafica, obtenemos un circulo muy parecido al de la guia, esto nos indica que no vamos a poder separar a las clases
# con una unica recta. que es lo q representamos en la grafica tambien

#lo q probe fue ir cambiando la arquitectura, primero ejecute con 2,3,3,1, luego con 2,4,4,1 y luego con 2,5,5,1 pero no hubo cambios a la hora de 
# ver la tasa de aciertos y de errados.
#entonces probe combinaciones 2,6,6,6,1