import os

import requests
import json
from pyquery import PyQuery
from tqdm import tqdm
import numpy as np
import pandas as pd

response = requests.get('https://www.camara.cl/sala/doGet.asmx/getLegislaturas')
legislaturas = json.loads(response.text)

leg_id = 52
response = requests.get(f'https://www.camara.cl/sala/doGet.asmx/getSesiones?prmLegiId={leg_id}')
sesiones = json.loads(response.text)

sesion_ids = [sesion['Id'] for sesion in sesiones['data'][::-1]]

favor = []
contra = []
for sesion_id in tqdm(sesion_ids):
    # Get votaciones por sesion
    response = requests.get(f'https://www.camara.cl/sala/doGet.asmx/getVotacionesPorSesion?prmSesionId={sesion_id}')
    votaciones_sesion = json.loads(response.text)

    votaciones_ids = [vs['Id'] for vs in votaciones_sesion['data'][::-1]]
    for votacion_id in votaciones_ids:
        # Get votos
        response = requests.get(f'https://www.camara.cl/sala/views/VotacionDetalle.aspx?prmVotacionId={votacion_id}')
        pq = PyQuery(response.text)

        si = pq('div.row.detalle-si.encabezado + div.row').text().split('\n')
        no = pq('div.row.detalle-no.encabezado + div.row').text().split('\n')
        # No consideraremos abstenciones:
        # abst = pq('div.row.detalle-abs.encabezado + div.row').text().split('\n')

        favor.append(si)
        contra.append(no)

# Obtener nombres de congresistas
congresistas = sorted(list(set(sum(favor, []) + sum(contra, []))))
print(f'Congresistas encontrados: {len(congresistas)}')

# Eliminar congresista sin nombre
congresistas.remove('')

# Votaciones
print(f'Votaciones encontradas: {len(favor)}')

votos = []
for congresista in congresistas:
    votos.append([1 if congresista in votacion[0] else
                  0 if congresista in votacion[1] else
                  None
                  for votacion in zip(favor, contra)])

votos = np.array(votos)

votos_df = pd.DataFrame({'nombre': congresistas})
votos_df = pd.concat([votos_df, pd.DataFrame(votos)], axis=1)
votos_df.columns = ['nombre'] + [f'votacion_{k + 1}' for k in range(votos.shape[1])]

save_path = 'dtbip/data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

votos_df.to_csv(os.path.join(save_path, 'votos.csv'), index=False)
