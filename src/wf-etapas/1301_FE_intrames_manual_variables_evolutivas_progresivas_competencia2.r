#!/usr/bin/env Rscript
cat( "ETAPA  z1301_FE_intrames_manual.r  INIT\n")

# Workflow  Feature Engineering intrames manual artesanal

# inputs
#  * dataset
# output  
#   un dataset algo mas grande:
#     misma cantidad de registros
#     nuevos atributos construidos en forma artesanal y con mucho sufrimiento
#     generados en codigo R,  especificos para este dataset y clase

# limpio la memoria
rm(list = ls(all.names = TRUE)) # remove all objects
gc(full = TRUE, verbose= FALSE)  # garbage collection

require("data.table", quietly=TRUE)
require("yaml", quietly=TRUE)


#cargo la libreria
# args <- c( "~/labo2024ba" )
args <- commandArgs(trailingOnly=TRUE)
source( paste0( args[1] , "/src/lib/action_lib.r" ) )
#------------------------------------------------------------------------------

atributos_presentes <- function( patributos )
{
  atributos <- unique( patributos )
  comun <- intersect( atributos, colnames(dataset) )

  return(  length( atributos ) == length( comun ) )
}
#------------------------------------------------------------------------------
# Esta es la parte que los alumnos deben desplegar todo su ingenio
# Agregar aqui sus PROPIAS VARIABLES manuales

AgregarVariables_IntraMes <- function(dataset) {
  cat( "inicio AgregarVariables_IntraMes()\n")
  gc(verbose= FALSE)
  # INICIO de la seccion donde se deben hacer cambios con variables nuevas

  # Aqui debe usted agregar sus propias nuevas variables
  # Crear la variable mpasivos_margen_lag1 como un lag de orden 1 de mpasivos_margen

  if( atributos_presentes( c("ctrx_quarter_normalizado", "ctarjeta_visa_transacciones") ))
    dataset[, iter_1_var_625 := rowSums(cbind(ctrx_quarter_normalizado, ctarjeta_visa_transacciones), na.rm = TRUE)]

  if( atributos_presentes( c("ctrx_quarter_normalizado", "cpayroll_trx") ))
    dataset[, iter_1_var_614 := rowSums(cbind(ctrx_quarter_normalizado, cpayroll_trx), na.rm = TRUE)]

   if( atributos_presentes( c("vm_status01", "cpayroll_trx") ))
    dataset[, iter_1_var_644 := vm_status01 / cpayroll_trx]

  if( atributos_presentes( c("cpayroll_trx", "ctarjeta_visa_transacciones") ))
    dataset[, iter_1_var_69 := cpayroll_trx * ctarjeta_visa_transacciones]

  if( atributos_presentes( c("cpayroll_trx", "ctarjeta_visa_transacciones") ))
    dataset[, iter_1_var_285 := cpayroll_trx / ctarjeta_visa_transacciones]

  if( atributos_presentes( c("cpayroll_trx", "vmr_mpagominimo") ))
    dataset[, iter_1_var_288 := cpayroll_trx / vmr_mpagominimo]

  if( atributos_presentes( c("cpayroll_trx", "vmr_mpagominimo") ))
    dataset[, iter_1_var_15 := cpayroll_trx / vmr_mpagominimo]

  if( atributos_presentes( c("vm_status01", "tcallcenter") ))
    dataset[, iter_1_var_796 := rowSums(cbind(vm_status01, tcallcenter), na.rm = TRUE)]

  if( atributos_presentes( c("ctrx_quarter_normalizado", "tcallcenter") ))
    dataset[, iter_1_var_227 := ctrx_quarter_normalizado / tcallcenter]

  if( atributos_presentes( c("vm_status01", "tcallcenter") ))
   dataset[, iter_1_var_796 := rowSums(cbind(vm_status01, ctarjeta_visa_transacciones), na.rm = TRUE)]
  
  if( atributos_presentes( c("ctrx_quarter_normalizado", "cliente_edad") ))
    dataset[, iter_1_var_20 := ctrx_quarter_normalizado * cliente_edad]

  if( atributos_presentes( c("ctarjeta_visa_transacciones", "tcallcenter") ))
    dataset[, iter_1_var_227 := ctarjeta_visa_transacciones / tcallcenter]

  if( atributos_presentes( c("ctrx_quarter_normalizado", "vmr_mpagominimo") ))
    dataset[, iter_1_var_796 := rowSums(cbind(ctrx_quarter_normalizado, vmr_mpagominimo), na.rm = TRUE)]

  if( atributos_presentes( c("cpayroll_trx", "vm_status01") ))
    dataset[, iter_1_var_227 := cpayroll_trx / vm_status01]

  if( atributos_presentes( c("tcallcenter", "vmr_mpagominimo") ))
    dataset[, iter_1_var_548 := tcallcenter / vmr_mpagominimo]

  if( atributos_presentes( c("ctrx_quarter_normalizado", "vmr_mpagominimo") ))
    dataset[, iter_1_var_18 := ctrx_quarter_normalizado / vmr_mpagominimo]

  if( atributos_presentes( c("vm_status01", "tcallcenter") ))
    dataset[, iter_1_var_487 := vm_status01 / tcallcenter] 

  if( atributos_presentes( c("cpayroll_trx", "vmr_mpagominimo") ))
    dataset[, iter_1_var_796 := rowSums(cbind(cpayroll_trx, vmr_mpagominimo), na.rm = TRUE)]

  if( atributos_presentes( c("ctrx_quarter_normalizado", "ctarjeta_visa_transacciones", "mpayroll_sobre_edad") ))
    dataset[, iter_2_var_617 := rowSums(cbind(ctrx_quarter_normalizado, ctarjeta_visa_transacciones, mpayroll_sobre_edad), na.rm = TRUE)]

  if( atributos_presentes( c("ctrx_quarter", "ctarjeta_visa_transacciones", "cpayroll_trx", "vmr_mpagominimo") ))
    dataset[, iter_2_var_667 := rowSums(cbind(ctrx_quarter, ctarjeta_visa_transacciones, cpayroll_trx / vmr_mpagominimo), na.rm = TRUE)]

   if( atributos_presentes( c("mprestamos_personales_rank", "mcaja_ahorro_rank") ))
    dataset[, iter_2_var_669 := rowSums(cbind(mprestamos_personales_rank, mcaja_ahorro_rank), na.rm = TRUE)]

  if( atributos_presentes( c("mprestamos_personales_rank", "mpasivos_margen_rank") ))
    dataset[, iter_2_var_674 := rowSums(cbind(mprestamos_personales_rank, mpasivos_margen_rank), na.rm = TRUE)]

  if( atributos_presentes( c("mtarjeta_visa_consumo_rank", "mcuenta_debitos_automaticos_rank") ))
    dataset[, iter_2_var_774 := rowSums(cbind(mtarjeta_visa_consumo_rank, mcuenta_debitos_automaticos_rank), na.rm = TRUE)]

  if( atributos_presentes( c("mprestamos_personales_rank", "mrentabilidad_annual_rank") ))
    dataset[, iter_2_var_673 := rowSums(cbind(mprestamos_personales_rank, mrentabilidad_annual_rank), na.rm = TRUE)]

  if( atributos_presentes( c("mcaja_ahorro_rank", "cpayroll_trx", "ctarjeta_visa_transacciones") ))
    dataset[, iter_2_var_83 := mcaja_ahorro_rank * cpayroll_trx * ctarjeta_visa_transacciones]

  if( atributos_presentes( c("mcuenta_corriente_rank", "mtarjeta_visa_consumo_rank") ))
    dataset[, iter_2_var_481 := mcuenta_corriente_rank / mtarjeta_visa_consumo_rank ]

  if( atributos_presentes( c("mcaja_ahorro_rank", "cpayroll_trx", "vmr_mpagominimo") ))
    dataset[, iter_2_var_78 := rowSums(cbind(cpayroll_trx, vmr_mpagominimo), na.rm = TRUE) * mcaja_ahorro_rank]



  # valvula de seguridad para evitar valores infinitos
  # paso los infinitos a NULOS
  infinitos <- lapply(
    names(dataset),
    function(.name) dataset[, sum(is.infinite(get(.name)))]
  )

  infinitos_qty <- sum(unlist(infinitos))
  if (infinitos_qty > 0) {
    cat(
      "ATENCION, hay", infinitos_qty,
      "valores infinitos en tu dataset. Seran pasados a NA\n"
    )
    dataset[mapply(is.infinite, dataset)] <<- NA
  }


  # valvula de seguridad para evitar valores NaN  que es 0/0
  # paso los NaN a 0 , decision polemica si las hay
  # se invita a asignar un valor razonable segun la semantica del campo creado
  nans <- lapply(
    names(dataset),
    function(.name) dataset[, sum(is.nan(get(.name)))]
  )

  nans_qty <- sum(unlist(nans))
  if (nans_qty > 0) {
    cat(
      "ATENCION, hay", nans_qty,
      "valores NaN 0/0 en tu dataset. Seran pasados arbitrariamente a 0\n"
    )

    cat("Si no te gusta la decision, modifica a gusto el programa!\n\n")
    dataset[mapply(is.nan, dataset)] <<- 0
  }

  cat( "fin AgregarVariables_IntraMes()\n")
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Aqui comienza el programa
cat( "ETAPA  z1301_FE_intrames_manual.r  START\n")
action_inicializar() 


# cargo el dataset donde voy a entrenar
# esta en la carpeta del exp_input y siempre se llama  dataset.csv.gz
# cargo el dataset
envg$PARAM$dataset <- paste0( "./", envg$PARAM$input, "/dataset.csv.gz" )
envg$PARAM$dataset_metadata <- read_yaml( paste0( "./", envg$PARAM$input, "/dataset_metadata.yml" ) )

cat( "lectura del dataset\n")
action_verificar_archivo( envg$PARAM$dataset )
cat( "Iniciando lectura del dataset\n" )
dataset <- fread(envg$PARAM$dataset)
cat( "Finalizada lectura del dataset\n" )

GrabarOutput()

# Agrego las variables manuales
cat( "variables intra mest\n")
AgregarVariables_IntraMes(dataset)

#------------------------------------------------------------------------------
# grabo el dataset
cat( "grabado del dataset\n")
cat( "Iniciando grabado del dataset\n" )
fwrite(dataset,
  file = "dataset.csv.gz",
  logical01 = TRUE,
  sep = ","
)
cat( "Finalizado grabado del dataset\n" )


# copia la metadata sin modificar
cat( "grabado de metadata\n")
write_yaml( envg$PARAM$dataset_metadata, 
  file="dataset_metadata.yml" )

#------------------------------------------------------------------------------

# guardo los campos que tiene el dataset
tb_campos <- as.data.table(list(
  "pos" = 1:ncol(dataset),
  "campo" = names(sapply(dataset, class)),
  "tipo" = sapply(dataset, class),
  "nulos" = sapply(dataset, function(x) {
    sum(is.na(x))
  }),
  "ceros" = sapply(dataset, function(x) {
    sum(x == 0, na.rm = TRUE)
  })
))

fwrite(tb_campos,
  file = "dataset.campos.txt",
  sep = "\t"
)

#------------------------------------------------------------------------------
cat( "Fin del programa\n")

envg$OUTPUT$dataset$ncol <- ncol(dataset)
envg$OUTPUT$dataset$nrow <- nrow(dataset)
envg$OUTPUT$time$end <- format(Sys.time(), "%Y%m%d %H%M%S")
GrabarOutput()

#------------------------------------------------------------------------------
# finalizo la corrida
#  archivos tiene a los files que debo verificar existen para no abortar

action_finalizar( archivos = c("dataset.csv.gz","dataset_metadata.yml")) 
cat( "ETAPA  z1301_FE_intrames_manual.r  END\n")
