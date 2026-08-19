def entrenar_neurona(w, epocas, df, nformula):
    historial_w = []
    historial_errores = []
    for epoca in range(epocas):
        errores = 0
        for i in range(len(df)):
            x1 = df.iloc[i, 0]
            x2 = df.iloc[i, 1]
            yd = df.iloc[i, 2]
            x = [-1, x1, x2] #el -1 es el bias q m dijo el profe (preg si ta bien inicializarlo asi)
           #valor q obtengo del prod interno es la formula del video
            v = 0
            for j in range(len(w)):
                v=v + w[j] * x[j]
            #sgn
            if v>=0:
                y=1
            else:
                y=-1
            # cuento errores
            if y!=yd:
                errores=errores + 1
            # adaptacion de pesos
            for j in range(len(w)):
                w[j] = w[j] + (nformula / 2) * (yd - y) * x[j]
        historial_w.append(w.copy()) #los guardo aca xq en el ej 2 tengo q usarlos p graficar
        # guardamos los errores de esta epoca
        historial_errores.append(errores)
        #antes mi funcion solo devolvia los ultimos w y eso no seria correcto
        #seria la tasa de aprendizaje
        print("epoca:", epoca + 1, "errores:", errores)
        #criterio de finalizacion
        if errores==0:
            break
    return historial_w, w, historial_errores
#preguntar a que se referian en clases con tener un while e<emax o algo asi y para que serviria armarlo asi