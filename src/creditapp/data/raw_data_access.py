import os
import pandas as pd
import numpy as np
from datetime import date

from src.creditapp.data.utils import datasets_paths

def get_credit_record():
    # Read credit record CSV file provided.
    df_credit_record = pd.read_csv(datasets_paths['credit_record'], 
                                   sep=',', decimal=',', encoding='utf-8')
    
    return df_credit_record

def get_application_record():
    # Read application record CSV file provided.
    df_clients = pd.read_csv(datasets_paths['application_record'], 
                             sep=',', decimal=',', encoding='utf-8')
    
    # Limpieza del dataset
    df_clients.columns = df_clients.columns.str.lower()
    df_clients.drop_duplicates(subset='id', keep='first', inplace=True)

    # Cálculo de la edad en base a la diferencia de días con el nacimineto
    df_clients.days_birth = abs(df_clients.days_birth)
    df_clients.loc[::, 'age'] = df_clients.days_birth // 365

    df_clients.drop('days_birth', axis=1, inplace=True)

    # Variable "days_employed"
    # La variable posee un 15% de su distribución con valores positivos. Debido a la falta de interpretación lógica de los mismos, 
    # se decide colocar nulls cuando esta feature toma un valor mayor a cero.

    mask = df_clients.days_employed > 0

    df_clients.loc[mask, 'days_employed'] = np.nan

    # Para reducir la "variabilidad" se hace una conversión de días a meses

    df_clients.loc[::, 'days_employed'] = abs(df_clients['days_employed'])
    df_clients.loc[::, 'days_employed'] = round(df_clients.days_employed / 30)
    df_clients.rename(columns={'days_employed': 'months_employed'}, inplace=True)
    
    # Para el armado del target se toma el criterio (preguntar si no es claro):
    #      1. Se iguala a 0 aquellos resúmenes de cuenta con C o X en su status (no debe)
    #      2. De esta forma nos queda una escala del 0 al 6 en el status de la cuenta (donde 0 es "no debe", el resto "debe")
    #      3. Se saca la media de estos "status" de cuenta.
    #      4. Mergeo de los datasets.
    #      5. Debido a la detección de filas duplicadas, se decide sacar la "media de medias" de los status de cuenta.
    df_account_status = get_credit_record()
    
    df_account_status.columns = df_account_status.columns.str.lower()
    df_account_status.drop('months_balance', axis=1, inplace=True)

    mask = (df_account_status.status == 'X') | (df_account_status.status == 'C')

    df_account_status.loc[mask, 'status'] = '-1'
    df_account_status['status'] = df_account_status['status'].astype('int64')

    df_account_status.loc[::, 'status'] = df_account_status.status + 1
    
    # Agrupado por id del estado de cuenta para armado del target
    df_grouped = df_account_status.groupby(['id']).mean()
    df_grouped.reset_index(inplace=True)
    
    df_merged = df_clients.merge(df_grouped, left_on='id', right_on='id', how='inner')
    
    df_aux = df_merged.copy()

    # Agrupado por columnas repetidas (aquellas que obviamente se repetian para diferentes id's
    df_aux.drop('id', axis=1, inplace=True)
    cols = ['code_gender', 'flag_own_car', 'flag_own_realty', 'cnt_children', 'amt_income_total', 'name_income_type', 
            'name_education_type', 'name_family_status', 'name_housing_type', 'months_employed', 'flag_mobil', 'flag_work_phone', 
            'flag_email', 'cnt_fam_members', 'age']

    df_grouped = df_aux.groupby(cols).status.mean()
    df_grouped = df_grouped.to_frame()
    
    df_merged.set_index(cols, inplace=True)
    df_merged['status_avg'] = df_grouped['status']
    df_merged.reset_index(inplace=True)
    
    # Fabricación manual del target, tomando como criterio una "media de medias" mayor o igual a 1
    df_merged.drop_duplicates(subset=cols, keep='first', inplace=True)
    df_merged.drop('status', axis=1, inplace=True)

    df_clients = df_merged.copy()

    mask = df_clients.status_avg >= 1

    df_clients.loc[::, 'target'] = 0
    df_clients.loc[mask, 'target'] = 1

    df_clients.drop('status_avg', axis=1, inplace=True)
    
    df_clients.loc[::, 'amt_income_total'] = df_clients['amt_income_total'].astype('float64')
    df_clients.loc[::, 'cnt_fam_members'] = df_clients['cnt_fam_members'].astype('float64')
    
    return df_clients
