from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Lista de nombres de variables
variable_names = ['Sexo', 'Edad', 'Estado Civil', 'Número de hijos', 'Nivel Educativo', 
                  'Miembros Familia', 'Renta Generada', 'Bienes en propiedad', 'Ahorros', 
                  'Gastos esenciales', 'Otros gastos', 'Asalariado', 
                  'Ppal fuente ingresos: granja propia', 'Ppal fuente ingresos: Negocios', 'Ppal fuente ingresos: Otros', 
                  'Ingresos agrícolas', 'Gastos corrientes agrícolas', 'Trabajador por Cuenta Ajena', 
                  'Inversiones Permanentes', 'Inversiones Temporales']

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    prediccion = None
    if request.method == 'POST':
        variables = [request.form.get(name) for name in variable_names]
        row_dict = pd.DataFrame([variables], columns=variable_names).iloc[0].to_dict()
        # Procesa los datos como lo hiciste en tu función procesar_datos
        row_dict2, prediccion = procesar_datos(*variables)
        resultado = row_dict

    return render_template('index.html', variable_names=variable_names, resultado=resultado, prediccion=prediccion)

def procesar_datos(*variables):
    cols = ['sex', 'Age', 'Married', 'Number_children', 'education_level',
            'total_members', 'gained_asset', 'durable_asset', 'save_asset',
            'living_expenses', 'other_expenses', 'incoming_salary',
            'incoming_own_farm', 'incoming_business', 'incoming_no_business',
            'incoming_agricultural', 'farm_expenses', 'labor_primary',
            'lasting_investment', 'no_lasting_investmen']
    
    df = pd.DataFrame([variables], columns=cols)

    # Hacemos las transformaciones necesarias para que los datos tengan el mismo formato que los de entrenamiento
    df['sex'] = df['sex'].replace({"Hombre": '0', "Mujer": '1'})
    df['Married'] = df['Married'].replace({"Soltero/a": '0', "Casado/a": '1'})
    df['incoming_salary'] = df['incoming_salary'].replace({"No": '0', "Sí": '1'})
    df['incoming_own_farm'] = df['incoming_own_farm'].replace({"No": '0', "Sí": '1'})
    df['incoming_business'] = df['incoming_business'].replace({"No": '0', "Sí": '1'})
    df['incoming_no_business'] = df['incoming_no_business'].replace({"No": '0', "Sí": '1'})
    df['labor_primary'] = df['labor_primary'].replace({"No": '0', "Sí": '1'})

    num_cols = ['Age', 'gained_asset', 'durable_asset', 'save_asset','living_expenses', 
                'other_expenses', 'incoming_agricultural', 'farm_expenses', 
                'lasting_investment', 'no_lasting_investmen']
    for col in num_cols:
        df[col] = df[col].astype(float)
    
    with open('col_transformer.pkl', 'rb') as f:
        col_transformer = pickle.load(f)
    
    X_transformed = col_transformer.transform(df)
    X_transformed_df = pd.DataFrame(X_transformed, dtype=np.float32)

    with open('modelo_entrenado.pkl', 'rb') as archivo:
        modelo_final = pickle.load(archivo)
    
    prediccion = modelo_final.predict(X_transformed_df)

    if prediccion[0] == 0:
        prediccion = 'Negativo'
    else:
        prediccion = 'Positivo'
    
    # Convertir la primera fila del DataFrame en un diccionario
    row_dict = df.iloc[0].to_dict()

    return row_dict, prediccion

@app.route('/leyenda')
def leyenda():
    return render_template('leyenda.html')

if __name__ == '__main__':
    app.run(debug=True)
