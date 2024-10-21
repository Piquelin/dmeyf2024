# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:06:29 2024

@author: jfgonzalez
"""
import re
import numpy as np
import polars as pl
import time


# Inicia el temporizador
start_time = time.time()


# %%


archivo_datos = '../datasets/competencia_02.parquet'

# # creo la carpeta donde va el experimento
# dir.create("./exp/", showWarnings = FALSE)
# dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# # Establezco el Working Directory DEL EXPERIMENTO
# setwd(paste0("./exp/", PARAM$experimento, "/"))

# %% PARAM

PARAM = {}

PARAM['experimento'] = "PP02723_us_ft_025"
PARAM['input'] = {
    'dataset': "./datasets/competencia_01.csv"
}

# Lugar para alternar semillas
# 799891, 799921, 799961, 799991, 800011
PARAM['semilla_azar'] = 799891  # Aquí poner su primer semilla

PARAM['driftingcorreccion'] = "ninguno"
PARAM['clase_minoritaria'] = ["BAJA+1", "BAJA+2"]

# Los meses en los que vamos a entrenar
PARAM['trainingstrategy'] = {
    'testing': [202104],
    'validation': [202103],
    'training': [202102, 202101],
    'final_train': [202104, 202103, 202102],
    'future': [202106],
    'training_undersampling': 0.25,
    'finaltrain_undersampling': 0.25
}

antes_de_leer = time.time()
print('antes_de_leer', antes_de_leer-start_time )
competencia_02 = pl.read_parquet(archivo_datos)


# %% funciones

def busco_inflacion(ultimo_mes='10_24', inicio="2019-01-01", fin="2021-08-01"):
    '''
    ultimo mes: MM_AA
    '''
    PATH = ('https://www.indec.gob.ar/ftp/cuadros/economia/' +
            f'sh_ipc_{ultimo_mes}.xls')

    ipc_nac = pl.read_excel(PATH, sheet_name='Variación mensual IPC Nacional',
                            read_options={"skip_rows": 0, "header_row":4,
                                          "n_rows":30, } )

    new_columns = ipc_nac.row(0)  # primera fila como encabezado
    # ipc_nac = ipc_nac.slice(2)  # eliminar la primera fila

    meses = pl.DataFrame(pl.Series(new_columns[1:])
                         .str.strptime(pl.Date, "%F %T", strict=False))

    ipc_df = pl.concat([meses, ipc_nac[3,1:].transpose()], how='horizontal')
    ipc_df.columns = (['mes', 'IPC'])



    # Filtrar el rango de fechas
    ipc_df = ipc_df.filter(
        (pl.col('mes') >= pl.lit(inicio).str.strptime(pl.Date, "%Y-%m-%d")) &
        (ipc_df['mes'] <= pl.lit(fin).str.strptime(pl.Date, "%Y-%m-%d"))
    )


    return ipc_df


def obtener_campos_monetarios(dataset):
    # Obtener los nombres de las columnas del dataset
    columnas = dataset.columns  # o dataset if ya es una lista de columnas

    # Definir la expresión regular
    pattern = r"^(m|Visa_m|Master_m|vm_m)"

    # Filtrar las columnas que coinciden con la expresión regular
    campos_monetarios = [col for col in columnas if re.match(pattern, col)]

    return campos_monetarios


def corregir_rotas(df):  # AKA AsignarNA_campomeses (catástrofe)
    # Obtiene la lista de meses únicos
    meses = df['foto_mes'].unique().to_list()

    # Recorre los meses
    listadfs = []
    for mes in meses:
        df_mes = df.filter(pl.col("foto_mes") == mes)

        # Verifica qué columnas tienen todos ceros excepto 'foto_mes' y 'clase_ternaria'
        todo_cero_mask = df_mes.select(pl.all().exclude("foto_mes", "clase_ternaria").cast(pl.Float64).sum() == 0).row(0)
        todo_cero_mask = (False,) + todo_cero_mask + (False,)
        cols_cero = [col for col, is_cero in zip(df_mes.columns, todo_cero_mask) if is_cero]

        # Asigna None a esas columnas
        df_mes = df_mes.with_columns([pl.when(pl.col(col)
                                              .is_not_null()).then(None)
                                      .otherwise(pl.col(col))
                                      .alias(col) for col in cols_cero])

        print('del mes', mes, 'reemplazamos las columnas', cols_cero)
        listadfs.append(df_mes)

    # Concatenar todos los dataframes de meses
    resultado_final = pl.concat(listadfs)
    return resultado_final


def arreglo_infinitos(polars_dataset):
    columns = obtener_campos_monetarios(polars_dataset)

    columns_with_infinity = [
        col for col in columns if polars_dataset
        .select(pl.col(col).is_infinite().any()).to_numpy()[0][0]
    ]
    print('Encontre estas columnas con infinitos:\n', columns_with_infinity)

    # Reemplazar valores infinitos por null en cada columna
    polars_dataset = polars_dataset.with_columns([
        pl.when(pl.col(col).is_infinite())
        .then(None)  # Reemplaza infinitos por None (null en Polars)
        .otherwise(pl.col(col))
        .alias(col)
        for col in columns_with_infinity
    ])
    return polars_dataset


# Suma de columnas (usando na.rm equivalente)
def sumas_max_min(competencia_02):
    # Suma de columnas y mínimo/máximo
    competencia_02 = competencia_02.with_columns([
        (pl.col("Master_mfinanciacion_limite") + pl.col("Visa_mfinanciacion_limite")).alias("vm_mfinanciacion_limite"),
        pl.min_horizontal([pl.col("Master_Fvencimiento"), pl.col("Visa_Fvencimiento")]).alias("vm_Fvencimiento"),
        pl.min_horizontal([pl.col("Master_Finiciomora"), pl.col("Visa_Finiciomora")]).alias("vm_Finiciomora"),
        (pl.col("Master_msaldototal") + pl.col("Visa_msaldototal")).alias("vm_msaldototal"),
        (pl.col("Master_msaldopesos") + pl.col("Visa_msaldopesos")).alias("vm_msaldopesos"),
        (pl.col("Master_msaldodolares") + pl.col("Visa_msaldodolares")).alias("vm_msaldodolares"),
        (pl.col("Master_mconsumospesos") + pl.col("Visa_mconsumospesos")).alias("vm_mconsumospesos"),
        (pl.col("Master_mconsumosdolares") + pl.col("Visa_mconsumosdolares")).alias("vm_mconsumosdolares"),
        (pl.col("Master_mlimitecompra") + pl.col("Visa_mlimitecompra")).alias("vm_mlimitecompra"),
        (pl.col("Master_madelantopesos") + pl.col("Visa_madelantopesos")).alias("vm_madelantopesos"),
        (pl.col("Master_madelantodolares") + pl.col("Visa_madelantodolares")).alias("vm_madelantodolares"),
        pl.max_horizontal([pl.col("Master_fultimo_cierre"), pl.col("Visa_fultimo_cierre")]).alias("vm_fultimo_cierre"),
        (pl.col("Master_mpagado") + pl.col("Visa_mpagado")).alias("vm_mpagado"),
        (pl.col("Master_mpagospesos") + pl.col("Visa_mpagospesos")).alias("vm_mpagospesos"),
        (pl.col("Master_mpagosdolares") + pl.col("Visa_mpagosdolares")).alias("vm_mpagosdolares"),
        pl.max_horizontal([pl.col("Master_fechaalta"), pl.col("Visa_fechaalta")]).alias("vm_fechaalta"),
        (pl.col("Master_mconsumototal") + pl.col("Visa_mconsumototal")).alias("vm_mconsumototal"),
        (pl.col("Master_cconsumos") + pl.col("Visa_cconsumos")).alias("vm_cconsumos"),
        (pl.col("Master_cadelantosefectivo") + pl.col("Visa_cadelantosefectivo")).alias("vm_cadelantosefectivo"),
        (pl.col("Master_mpagominimo") + pl.col("Visa_mpagominimo")).alias("vm_mpagominimo")
    ])
    return competencia_02


def ratios_varios(competencia_02):
    competencia_02 = competencia_02.with_columns([
        (pl.col("Master_mlimitecompra") / pl.col("vm_mlimitecompra")).alias("vmr_Master_mlimitecompra"),
        (pl.col("Visa_mlimitecompra") / pl.col("vm_mlimitecompra")).alias("vmr_Visa_mlimitecompra"),
        (pl.col("vm_msaldototal") / pl.col("vm_mlimitecompra")).alias("vmr_msaldototal"),
        (pl.col("vm_msaldopesos") / pl.col("vm_mlimitecompra")).alias("vmr_msaldopesos"),
        (pl.col("vm_msaldopesos") / pl.col("vm_msaldototal")).alias("vmr_msaldopesos2"),
        (pl.col("vm_msaldodolares") / pl.col("vm_mlimitecompra")).alias("vmr_msaldodolares"),
        (pl.col("vm_msaldodolares") / pl.col("vm_msaldototal")).alias("vmr_msaldodolares2"),
        (pl.col("vm_mconsumospesos") / pl.col("vm_mlimitecompra")).alias("vmr_mconsumospesos"),
        (pl.col("vm_mconsumosdolares") / pl.col("vm_mlimitecompra")).alias("vmr_mconsumosdolares"),
        (pl.col("vm_madelantopesos") / pl.col("vm_mlimitecompra")).alias("vmr_madelantopesos"),
        (pl.col("vm_madelantodolares") / pl.col("vm_mlimitecompra")).alias("vmr_madelantodolares"),
        (pl.col("vm_mpagado") / pl.col("vm_mlimitecompra")).alias("vmr_mpagado"),
        (pl.col("vm_mpagospesos") / pl.col("vm_mlimitecompra")).alias("vmr_mpagospesos"),
        (pl.col("vm_mpagosdolares") / pl.col("vm_mlimitecompra")).alias("vmr_mpagosdolares"),
        (pl.col("vm_mconsumototal") / pl.col("vm_mlimitecompra")).alias("vmr_mconsumototal"),
        (pl.col("vm_mpagominimo") / pl.col("vm_mlimitecompra")).alias("vmr_mpagominimo"),
        (pl.col("mpayroll") / pl.col("cliente_edad")).alias("mpayroll_sobre_edad")  # Calcular mpayroll_sobre_edad
    ])

    arreglo_infinitos(competencia_02)
    return competencia_02


def lags_y_deltalags(dataset, campitos = ["numero_de_cliente", "foto_mes", "clase_ternaria"]):
    # campitos =  Columnas no lagueables
    # Columnas a las que les aplicaremos los lags
    cols_lagueables = [col for col in dataset.columns if col not in campitos]

    # Filtrar columnas de tipo string
    string_cols = [col for col in dataset.columns if dataset[col].dtype == pl.Utf8]
    # Excluir columnas de tipo string de cols_lagueables
    cols_lagueables = [col for col in cols_lagueables if col not in string_cols]

    # Ordenamos el dataset por numero_de_cliente y foto_mes
    dataset = dataset.sort(["numero_de_cliente", "foto_mes"])

    # Ordenamos el dataset por numero_de_cliente y foto_mes
    dataset = dataset.sort(["numero_de_cliente", "foto_mes"])

    # Creamos los lags de orden 1
    dataset = dataset.with_columns([
        pl.col(col).shift(1).over("numero_de_cliente").alias(f"{col}_lag1")
        for col in cols_lagueables
    ])

    # Agregamos los delta lags de orden 1
    dataset = dataset.with_columns([
        (pl.col(col) - pl.col(f"{col}_lag1")).alias(f"{col}_delta1")
        for col in cols_lagueables
    ])
    return dataset


#  part_future, part_validation, part_testing y part_training
def armo_particiones(dataset, PARAM):
    # Inicializamos con 0
    dataset = dataset.with_columns([
        pl.lit(0).alias("part_future"),
        pl.lit(0).alias("part_validation"),
        pl.lit(0).alias("part_testing"),
        pl.lit(0).alias("part_training"),
        pl.lit(0).alias("part_final_train")
    ])

    # Actualizamos las columnas según las condiciones dadas
    dataset = dataset.with_columns([
        pl.when(pl.col("foto_mes").is_in(PARAM["trainingstrategy"]["future"]))
        .then(1)
        .otherwise(pl.col("part_future")).alias("part_future"),

        pl.when(pl.col("foto_mes").is_in(PARAM["trainingstrategy"]["validation"]))
        .then(1)
        .otherwise(pl.col("part_validation")).alias("part_validation"),

        pl.when(pl.col("foto_mes").is_in(PARAM["trainingstrategy"]["testing"]))
        .then(1)
        .otherwise(pl.col("part_testing")).alias("part_testing")
    ])

    # Generar una columna de azar (números aleatorios)
    np.random.seed(PARAM["semilla_azar"])
    dataset = dataset.with_columns([
        pl.lit(np.random.uniform(size=dataset.shape[0])).alias("azar")
    ])

    # Crear la columna part_training con condiciones adicionales
    dataset = dataset.with_columns([
        pl.when(
            (pl.col("foto_mes").is_in(PARAM["trainingstrategy"]["training"])) &
            (
                (pl.col("azar") <= PARAM["trainingstrategy"]["training_undersampling"]) |
                (pl.col("clase_ternaria").is_in(PARAM["clase_minoritaria"]))
            )
        ).then(1).otherwise(0).alias("part_training")
    ])

    # Crear la columna part_final_train con condiciones adicionales
    dataset = dataset.with_columns([
        pl.when(
            (pl.col("foto_mes").is_in(PARAM["trainingstrategy"]["final_train"])) &
            (
                (pl.col("azar") <= PARAM["trainingstrategy"]["finaltrain_undersampling"]) |
                (pl.col("clase_ternaria").is_in(PARAM["clase_minoritaria"]))
            )
        ).then(1).otherwise(0).alias("part_final_train")
    ])

    # Eliminar la columna "azar" que ya no se usa
    dataset = dataset.drop("azar")

    return dataset


def ctrx_quarter_normalizado(competencia_02):
    # Aplicar las condiciones
    competencia_02 = competencia_02.with_columns(
        pl.when(pl.col("cliente_antiguedad") == 1)
        .then(pl.col("ctrx_quarter") * 3)
        .when(pl.col("cliente_antiguedad") == 2)
        .then(pl.col("ctrx_quarter") * 1.5)
        .otherwise(pl.col("ctrx_quarter"))
        .alias("ctrx_quarter_normalizado")
    )
    return competencia_02

# %% listas varias

# busco_inflacion()

# %%
# valores financieros
# # meses que me interesan
# vfoto_mes = [202101, 202102, 202103, 202104, 202105, 202106]

# # los valores que siguen fueron calculados por alumnos
# #  si no esta de acuerdo, cambielos por los suyos

# # momento 1.0  31-dic-2020 a las 23:59
# vIPC [ 0.9680542110, 0.9344152616, 0.8882274350,
#       0.8532444140, 0.8251880213, 0.8003763543]

# vdolar_blue = [157.900000, 149.380952, 143.615385,
#                146.250000, 153.550000, 162.000000]


# vdolar_oficial = [ 91.474000,  93.997778,  96.635909,
#                   98.526000,  99.613158, 100.619048]

# vUVA = [  0.9669867858358365, 0.9323750098728378, 0.8958202912590305,
#         0.8631993702994263, 0.8253893405524657, 0.7928918905364516 ]


# %% corrigo caatstrofes
antes_de_correcciones = time.time()
print('antes_de_correcciones', antes_de_correcciones-start_time )
competencia_02 = corregir_rotas(competencia_02)

# %% aqui aplico un metodo para atacar el data drifting
# hay que probar experimentalmente cual funciona mejor

# %% concept drifting ??

# datos con concept driftiing, ver si los sacamos
# dataset[, cprestamos_personales := NULL ]
# dataset[, mprestamos_personales := NULL ]
# dataset[, cplazo_fijo := NULL ]
# dataset[, ctarjeta_debito := NULL ]

# %% ctrx_quarter_normalizado

ctrx_quarter_normalizado(competencia_02)

competencia_02 = sumas_max_min(competencia_02)
competencia_02 = ratios_varios(competencia_02)

# %% lags_y_deltalags
antes_de_lags = time.time()
print('antes_de_lags', antes_de_lags-start_time )
competencia_var = lags_y_deltalags(competencia_02)

# %%

competencia_02 = armo_particiones(competencia_02, PARAM)
antes_de_guardar = time.time()
print('antes_de_guardar', antes_de_guardar-start_time )
competencia_02.write_parquet('../datasets/dataset02_pp_py.parquet')


# Calcula el tiempo de ejecución
end_time = time.time()
execution_time = end_time - start_time
print('execution_time', execution_time)
