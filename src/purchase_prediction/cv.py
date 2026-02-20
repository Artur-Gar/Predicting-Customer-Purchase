import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_curve
from sklearn.metrics import make_scorer

### Common CV & scorer
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_macro_scorer = make_scorer(f1_score, average='macro')

def GridSearchCV_fun(X_train, y_train, models_and_grids):
    best_models = {}
    optimal_thresholds = {}
    for model, param_grid, name in models_and_grids:
        grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=f1_macro_scorer,
        cv=cv,
        n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        cur_best_model = grid_search.best_estimator_
        best_models[name] = cur_best_model

        # Predict probabilities on training set to find optimal threshold
        if hasattr(cur_best_model, "predict_proba"):
            y_probs = cur_best_model.predict_proba(X_train)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_train, y_probs)
            optimal_threshold = thresholds[np.argmax(tpr - fpr)]  # criterion for optimal treshold
            optimal_thresholds[name] = optimal_threshold
            print(f"{name} Optimal Threshold: {optimal_threshold:.4f}\n")
        else:
            print(f"{name} does not support predict_proba, using default threshold of 0.5.\n")
            optimal_thresholds[name] = 0.5

        print(f"{name} CV best F1 (macro): {grid_search.best_score_:.4f}\n")

    return best_models, optimal_thresholds


def get_scores(X_test, y_test, best_models, optimal_thresholds):
    test_scores = {}
    best_score = 0
    best_model = ''
    for name, model in best_models.items():

        # try to utilise optimal_threshold info
        if hasattr(model, "predict_proba"):
            y_pred_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_probs >= optimal_thresholds[name]).astype(int)
        else:
            y_pred = model.predict(X_test)

        f1_macro = f1_score(y_test, y_pred, average='macro')
        test_scores[name] = f1_macro
        if f1_macro > best_score:
            best_score = f1_macro
            best_model = name
    
    return test_scores, best_model 
