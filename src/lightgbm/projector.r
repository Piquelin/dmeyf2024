# para correr el Google Cloud

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")


# defino los parametros de la corrida, en una lista, la variable global  PARAM
PARAM <- list()
PARAM$experimento <- "pronostico_1"

PARAM$semilla_primigenia <- 799891


PARAM$input$dataset <- "./datasets/competencia_02.csv"
PARAM$input$training <- c(202105) # meses donde se entrena el modelo
PARAM$input$future <- c(201901,201902,201903,201904,201905,201906,
                        201907,201908,201909,201910,201911,201912,
                        202001,202002,202003,202004,202005,202006,
                        202007,202008,202009,202010,202011,202012,
                        202101,202102,202103,202104,202105,202106,
                        202108,202108) # meses donde se aplica el modelo


PARAM$finalmodel$num_iterations <- 608
PARAM$finalmodel$learning_rate <- 0.0375106723386044
PARAM$finalmodel$feature_fraction <- 0.567454026661252
PARAM$finalmodel$min_data_in_leaf <- 789
PARAM$finalmodel$num_leaves <- 750
PARAM$finalmodel$verbosity <- -100.0
PARAM$finalmodel$force_row_wise <- FALSE


PARAM$finalmodel$max_bin <- 31

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui empieza el programa
# setwd("E:/Users/Piquelin/Documents/Maestría_DataMining/Economia_y_finanzas/")
setwd("C:/Users/jfgonzalez/Documents/Documentación_maestría/Economía_y_finanzas/")
# setwd("~/datasets/") # Establezco el Working Directory


# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)


#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
# set trabaja con la clase  POS = { BAJA+1, BAJA+2 }
# esta estrategia es MUY importante
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------


# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# creo las carpetas donde van los resultados
# creo la carpeta donde va el experimento
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))



# dejo los datos en el formato que necesita LightGBM
dtrain <- lgb.Dataset(
  data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
  label = dataset[train == 1L, clase01]
)

# genero el modelo
# estos hiperparametros  salieron de una laaarga Optmizacion Bayesiana
modelo <- lgb.train(
  data = dtrain,
  param = list(
    objective = "binary",
    max_bin = PARAM$finalmodel$max_bin,
    learning_rate = PARAM$finalmodel$learning_rate,
    num_iterations = PARAM$finalmodel$num_iterations,
    num_leaves = PARAM$finalmodel$num_leaves,
    min_data_in_leaf = PARAM$finalmodel$min_data_in_leaf,
    feature_fraction = PARAM$finalmodel$feature_fraction,
    verbosity = PARAM$finalmodel$verbosity,
    force_row_wise = PARAM$finalmodel$force_row_wise,
    seed = 799891
  )
)

#--------------------------------------
# ahora imprimo la importancia de variables
tb_importancia <- as.data.table(lgb.importance(modelo))
archivo_importancia <- "impo.txt"

fwrite(tb_importancia,
  file = archivo_importancia,
  sep = "\t"
)

#--------------------------------------
# grabo a disco el modelo en un formato para seres humanos ... ponele ...

lgb.save(modelo, "modelo.txt" )

#--------------------------------------

# aplico el modelo a los datos sin clase
dapply <- dataset[foto_mes == PARAM$input$future]

# aplico el modelo a los datos nuevos
prediccion <- predict(
  modelo,
  data.matrix(dapply[, campos_buenos, with = FALSE])
)

# genero la tabla de entrega
tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
tb_entrega[, prob := prediccion]


# ordeno por probabilidad descendente
setorder(tb_entrega, -prob)
 

# grabo las probabilidad del modelo
fwrite(tb_entrega,
  file = "prediccion.txt",
  sep = "\t"
)
# 
# 
# # genero archivos con los  "envios" mejores
# # suba TODOS los archivos a Kaggle
# 
# cortes <- seq(500, 20000, by = 500)
# for (envios in cortes) {
#   tb_entrega[, Predicted := 0L]
#   tb_entrega[1:envios, Predicted := 1L]
# 
#   fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
#     file = paste0(PARAM$experimento, "_", envios, ".csv"),
#     sep = ","
#   )
# }

cat("\n\nLa generacion de los archivos para Kaggle ha terminado\n")
