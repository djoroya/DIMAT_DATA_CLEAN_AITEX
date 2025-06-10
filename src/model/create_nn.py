from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

def create_nn(df,outvars, hidden_layers=(200, 100, 50), test_size=0.2, random_state=42):
    """
    Entrena un modelo de red neuronal para predecir Tg1, Tg2 y Tg3.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con las variables predictoras y Tg1, Tg2, Tg3.
        hidden_layers (tuple): Arquitectura de la red neuronal.
        test_size (float): Proporción del conjunto de prueba.
        random_state (int): Semilla aleatoria.
        
    Retorna:
        dict: MAPE para entrenamiento y test, por variable.
    """

    # Variables objetivo
    df = df.copy()
    y = df[outvars]
    X = df.drop(columns=outvars)
    
    # División en entrenamiento/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Escalado
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Modelo
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=2000, random_state=random_state)
    model.fit(X_train_scaled, y_train_scaled)

    # Predicción
    
    y_train_pred = scaler_y.inverse_transform(model.predict(X_train_scaled))
    y_test_pred = scaler_y.inverse_transform(model.predict(X_test_scaled))

    # MAPE
    mape_train = 100*mean_absolute_percentage_error(y_train, y_train_pred, multioutput='raw_values')
    mape_test = 100*mean_absolute_percentage_error(y_test, y_test_pred, multioutput='raw_values')

    return {
        "MAPE_train": dict(zip(outvars, mape_train)),
        "MAPE_test": dict(zip(outvars, mape_test)),
        "model": model,
        "y": y,
        "X": X,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "X_test":X_test
    }