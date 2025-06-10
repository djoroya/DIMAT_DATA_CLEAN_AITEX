
from functions.fitnn import fitnn

import optuna


def fin_best(df, inputs_list, outputs_list,init_epoch=2000):
    # Función objetivo para Optuna
    def objective(trial):
        # Sugerir valores para los hiperparámetros
        hidden_dim = trial.suggest_int('hidden_dim', 2, 6)
        neurons = trial.suggest_int('neurons', 10, 100)
        lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        results = fitnn(df, inputs_list, outputs_list, {
            "hidden": hidden_dim,
            "epochs": init_epoch,
            "neurons": neurons,
            "weight_decay": weight_decay,
            "l2_weight": weight_decay,
            "lr": lr,
        })

        return results["etest_mean"]


    # Crear un estudio de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)  # Número de pruebas

    # Mejores hiperparámetros encontrados
    print("Best hyperparameters:", study.best_params)
    best_params = study.best_params

    results_best = fitnn(df, inputs_list, outputs_list, {
    "hidden": best_params["hidden_dim"],
    "epochs": 20000,
    "neurons": best_params["neurons"],
    "weight_decay": best_params["weight_decay"],
    "l2_weight": best_params["weight_decay"],
    "lr": best_params["lr"],
    })

    return results_best
