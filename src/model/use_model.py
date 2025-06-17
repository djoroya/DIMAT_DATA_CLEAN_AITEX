# use model 
import joblib
import pandas as pd
import os

file_path = os.path.dirname(os.path.abspath(__file__)) 

nn_model = joblib.load(os.path.join(file_path, "model/nn_model.joblib"))
scaler_X = joblib.load(os.path.join(file_path, "model/nn_scaler_X.joblib"))
scaler_y = joblib.load(os.path.join(file_path, "model/nn_scaler_y.joblib"))

def predict_nn_onehot(Xdict):
    """
    Predict using the trained neural network model.
    
    'Nº holes / Nº filaments'
    'Diameter (mm)'
    'L/D'
    'Filter (um)'
    'Sand quantity (cm)'
    'Vpump (rpm)'
    'Vext (rpm)'
    'TE1 (°C)'
    'TE2 (°C)'
    'TE3 (°C)'
    'TE4 (°C)'
    'TD(°C)'
    'PD(bar)'
    'PE(bar)'
    'VTO (m/min)'
    'VG1(m/min)'
    'VG2(m/min)'
    'VG3(m/min)'
    'VG4(m/min)'
    'Vwinder (m/min)'
    'Tg1(°C)'
    'Tg2(°C)'
    'Tg3(°C)'
    'Quench (%)'
    'Spin. F (rmp)'
    'Height (m)'
    'Sand size min (um)'
    'Sand size max (um)'
    'Material_PA'
    'Material_PBT'
    'Material_PE'
    'Material_PES'
    'Material_PET'
    'Material_PHB'
    'Material_PLA'
    'Material_PP'
    'Material_TPE'
    'Material_TPX'
    
    Parameters:
    X (dict-like): Input features to predict.
    

    Returns:
    array: Predicted values.
    """
    
    # Convert input dict to DataFrame
    X = pd.DataFrame(Xdict)
    X_scaled = scaler_X.transform(X)
    y_pred_scaled = nn_model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    outvars = ["Count (dtex)", "Elongation (%)", "Tenacity (g/den)"]
    y_pred = pd.DataFrame(y_pred, columns=outvars)
    # 
    return y_pred

def predict_nn(samples):

    # 'Material_PA'   : 0.0,
    # 'Material_PBT'  : 0.0,
    # 'Material_PE'   : 0.0,
    # 'Material_PES ' : 0.0,
    # 'Material_PET'  : 1.0,
    # 'Material_PHB'  : 0.0,
    # 'Material_PLA'  : 0.0,
    # 'Material_PP'   : 0.0,
    # 'Material_TPE'  : 0.0,
    # 'Material_TPX'  : 0.0


    for sample in samples:

        materials_lib = ["PA" , "PBT", "PE", 
                        "PES", "PET", "PHB", 
                        "PLA", "PP" , "TPE", 
                        "TPX"]

        material = sample.pop('Material', None);

        material_onehot = {f'Material_{mat}': 1.0 if mat == material 
                        else 0.0 for mat in materials_lib}
        sample.update(material_onehot)

    return predict_nn_onehot(samples)