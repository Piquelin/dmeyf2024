# Arbol elemental con libreria  rpart
# Debe tener instaladas las librerias  data.table ,  rpart  y  rpart.plot

# cargo las librerias que necesito
require("data.table")
require("rpart")
require("rpart.plot")

# Aqui se debe poner la carpeta de la materia de SU computadora local
setwd("E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas") # Establezco el Working Directory

# cargo el dataset que tiene la clase calculada !
dataset <- fread("./datasets/competencia_01.csv")

dtrain <- dataset[foto_mes == 202104] # defino donde voy a entrenar
dapply <- dataset[foto_mes == 202106] # defino donde voy a aplicar el modelo

# genero el modelo,  aqui se construye el arbol
# quiero predecir clase_ternaria a partir de el resto de las variables
modelo <- rpart(
    formula = "clase_ternaria ~ .",
    data = dtrain, # los datos donde voy a entrenar
    xval = 0,
    cp = -0.15, # esto significa no limitar la complejidad de los splits
    minsplit = 1185, # minima cantidad de registros para que se haga el split
    minbucket = 150, # tamaño minimo de una hoja
    maxdepth = 7  # profundidad maxima del arbol
)


# grafico el arbol
prp(modelo,
    extra = 101, digits = -5,
    branch = 1, type = 4, varlen = 0, faclen = 0
)


# aplico el modelo a los datos nuevos
prediccion <- predict(
    object = modelo,
    newdata = dapply,
    type = "prob"
)

# prediccion es una matriz con TRES columnas,
# llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
# cada columna es el vector de probabilidades

# agrego a dapply una columna nueva que es la probabilidad de BAJA+2
dapply[, prob_baja2 := prediccion[, "BAJA+2"]]

# solo le envio estimulo a los registros
#  con probabilidad de BAJA+2 mayor  a  1/40
dapply[, Predicted := as.numeric(prob_baja2 > 1 / 40)]

# genero el archivo para Kaggle
# primero creo la carpeta donde va el experimento
dir.create("./exp/")
dir.create("./exp/KA2004")

# solo los campos para Kaggle
fwrite(dapply[, list(numero_de_cliente, Predicted)],
        file = "./exp/KA2004/118_150_7.csv",
        sep = ","
)



#--------------------------------------


# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, prob := prediccion[, "BAJA+2"]]

# grabo las probabilidad del modelo
fwrite(tb_entrega,
       file = "prediccion.txt",
       sep = "\t"
)

# ordeno por probabilidad descendente
setorder(tb_entrega, -prob)


# genero archivos con los  "envios" mejores
# suba TODOS los archivos a Kaggle

cortes <- seq(500, 20000, by = 500)
for (envios in cortes) {
  tb_entrega[, Predicted := 0L]
  tb_entrega[1:envios, Predicted := 1L]
  
  fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
         file = paste0(PARAM$experimento, "015_1185_150_7_", envios, ".csv"),
         sep = ","
  )
}

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")
