import os

import requests
import json
from pyquery import PyQuery
from tqdm import tqdm


response = requests.get('https://www.camara.cl/sala/doGet.asmx/getLegislaturas')
legislaturas = json.loads(response.text)

leg_id = 52
response = requests.get(f'https://www.camara.cl/sala/doGet.asmx/getSesiones?prmLegiId={leg_id}')
sesiones = json.loads(response.text)

sesion_ids = [sesion['Id'] for sesion in sesiones['data'][::-1]]

favor = []
contra = []
for sesion_id in tqdm(sesion_ids[:2]):
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
        #abst = pq('div.row.detalle-abs.encabezado + div.row').text().split('\n')

        favor.append(si)
        contra.append(no)

diputados = sorted(list(set(sum(favor, []) + sum(contra, []))))
len(diputados)

diputados

for diputado in diputados:
    print([diputado in votacion for votacion in favor])
