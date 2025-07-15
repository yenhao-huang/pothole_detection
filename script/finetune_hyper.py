import argparse
import shutil
import os
from ultralytics import YOLO
import optuna


def run_yolo_train(params, model_name, imgsz, trial_id, epochs):
    # Ensure checkpoint exists or download
    model_path_ckpt = f"ckpt/{model_name}.pt"
    model_path_local = f"./{model_name}.pt"

    if not os.path.exists(model_path_ckpt):
        model = YOLO(f"{model_name}.pt")
        if os.path.exists(model_path_local):
            os.makedirs('ckpt', exist_ok=True)
            shutil.move(model_path_local, model_path_ckpt)
    else:
        model = YOLO(model_path_ckpt)

    # Train model with trial-specific augmentation params
    model.train(
        data='data/pothole_data/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        project='hpo_trials',
        name=f'trial_{trial_id}',
        degrees=params['degrees'],
        translate=params['translate'],
        scale=params['scale'],
        shear=params['shear'],
        perspective=params['perspective'],
        flipud=0.0,
        fliplr=params['fliplr'],
        hsv_h=params['hsv_h'],
        hsv_s=params['hsv_s'],
        hsv_v=params['hsv_v'],
        mosaic=params['mosaic'],
        mixup=0.0,
        copy_paste=0.0,
        cache=True,
        verbose=False
    )

    # Validate and extract mAP50
    metrics = model.val()
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
    print(f"Trial {trial_id} mAP50: {map50}")
    return map50


def objective(trial, model_name, imgsz, epochs):
    params = {
        'degrees': trial.suggest_float('degrees', 0.0, 20.0),
        'translate': trial.suggest_float('translate', 0.0, 0.3),
        'scale': trial.suggest_float('scale', 0.0, 1.0),
        'shear': trial.suggest_float('shear', 0.0, 10.0),
        'perspective': trial.suggest_float('perspective', 0.0, 0.001),
        'fliplr': trial.suggest_float('fliplr', 0.0, 0.5),
        'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.05),
        'hsv_s': trial.suggest_float('hsv_s', 0.0, 0.9),
        'hsv_v': trial.suggest_float('hsv_v', 0.0, 0.9),
        'mosaic': trial.suggest_float('mosaic', 0.0, 1.0)
    }
    return run_yolo_train(params, model_name, imgsz, trial.number, epochs)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for YOLOv11.")
    parser.add_argument('--model', type=str, default="yolo11n", help='YOLO model name (e.g., yolov11n)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs per trial')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=6, help='Number of parallel trials')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, args.model, args.imgsz, args.epochs),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs  # Enable multi-threaded trials
    )

    print("Best hyperparameters:", study.best_params)
    print("Best mAP50:", study.best_value)

    # Save best hyperparameters to YAML
    os.makedirs('best_config', exist_ok=True)
    best_hyp = study.best_params
    with open('best_config/best_hyp.yaml', 'w') as f:
        yaml.dump(best_hyp, f)
    print("Saved best hyperparameters to best_config/best_hyp.yaml")

if __name__ == "__main__":
    main()