import os

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

# Consultar datos en sitio de bcn
sparql = SPARQLWrapper("http://datos.bcn.cl/sparql")
query = '''
    SELECT ?fecha ?nombre ?texto WHERE {
        ?participacion a bcnres:Participacion;
        bcnres:tieneTipoParticipacion ?tipo;
        bcnres:esParteDe ?documento;
        bcnres:tieneEmisor ?persona;
        rdf:value ?texto.

        ?documento dc:date ?fecha;
        bcncon:perteneceA ?legislatura.
        ?persona rdfs:label ?nombre.


        FILTER(?tipo= bcnres:Intervencion)
        FILTER(?legislatura= <http://datos.bcn.cl/recurso/cl/legislatura/368>)

    }
    '''

sparql.setQuery(query)
sparql.setReturnFormat(JSON)
print('Realizando consulta:\n'+query+'\nEn: http://datos.bcn.cl/sparql')
results = sparql.query().convert()

# Procesar resultado
vars_ = results['head']['vars']
results_len = len(results['results']['bindings'])
values = []
for var in vars_:
    values.append([results['results']['bindings'][i][var]['value']
                   if var in results['results']['bindings'][i].keys()
                   else float('nan')
                   for i in range(results_len)])

df = pd.DataFrame(values).T
df.columns = vars_

save_path = 'data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

df.to_csv(os.path.join(save_path, 'intervenciones.csv'), index=False)
