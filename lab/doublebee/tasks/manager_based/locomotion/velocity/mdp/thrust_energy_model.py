import json
import os

_DIR = os.path.dirname(__file__)
_DEFAULT_THRUST_PARAMS = os.path.join(_DIR, "pwm2thrust_params.json")
_DEFAULT_POWER_PARAMS = os.path.join(_DIR, "pwm2power_params.json")


def _default_params_path(target: str) -> str:
    """Default JSON path for a given target (thrust or power)."""
    if target == "power":
        return _DEFAULT_POWER_PARAMS
    return _DEFAULT_THRUST_PARAMS


def pwm_to_thrust(
    pwm: float | list[float],
    model_path: str | None = None,
    target: str = "thrust",
):
    """
    Infer thrust (N) or power (W) from PWM using the saved best-fit model.

    Call this for inference without refitting. Run PWM2Thrust_regression() once
    per target to fit and save the model; then use this function anywhere.

    Args:
        pwm: PWM value(s). Scalar or array-like.
        model_path: Path to params JSON. If None, uses default for target
            (pwm2thrust_params.json or pwm2power_params.json next to this file).
        target: "thrust" for PWM -> Thrust (N), "power" for PWM -> Power (W).

    Returns:
        Thrust in N or power in W. Same shape as pwm (scalar or array).

    Raises:
        FileNotFoundError: If no saved model exists at model_path.
        ValueError: If target is not "thrust" or "power".
    """
    import numpy as np

    if target not in ("thrust", "power"):
        raise ValueError(f"target must be 'thrust' or 'power', got {target!r}")
    path = model_path if model_path is not None else _default_params_path(target)
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    coeffs = np.array(params["coeffs"], dtype=float)
    is_exponential = params["is_exponential"]
    x = np.asarray(pwm, dtype=float)
    if is_exponential:
        out = np.exp(np.polyval(coeffs, x))
    else:
        out = np.polyval(coeffs, x)
    return float(out) if np.ndim(out) == 0 else out


def PWM2Thrust_regression(
    csv_path: str | None = None,
    save_path: str | None = None,
    target: str = "thrust",
) -> tuple:
    """
    Build a regression model to convert PWM to thrust (N) or power (W) from CSV data.

    Loads PWM vs Thrust/Power from .csv, fits linear and polynomial models,
    compares errors (RMSE), and returns the best model as a callable predict(pwm).

    Args:
        csv_path: Path to CSV with columns "PWM", "Thrust (N)", "Power (W)".
            If None, uses PWM2TE.csv next to this file.
        save_path: Path to save best model params (JSON). If None, uses
            pwm2thrust_params.json or pwm2power_params.json for the chosen target.
        target: "thrust" to regress PWM -> Thrust (N), "power" to regress PWM -> Power (W).

    Returns:
        Tuple of (predict_fn, best_model_name, best_rmse). predict_fn(pwm) returns
        thrust in N or power in W depending on target.
    """
    import csv
    import os
    import numpy as np

    if target not in ("thrust", "power"):
        raise ValueError(f"target must be 'thrust' or 'power', got {target!r}")

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "PWM2TE.csv")

    # Load data: column depends on target
    col = "Thrust (N)" if target == "thrust" else "Power (W)"
    unit = "N" if target == "thrust" else "W"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    pwm = np.array([float(r["PWM"]) for r in rows])
    y = np.array([float(r[col]) for r in rows])
    x = pwm

    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    results: list[tuple[str, np.ndarray, float]] = []  # (name, coeffs_or_None, rmse)

    # --- Linear: thrust = a * pwm + b ---
    coeffs_lin = np.polyfit(x, y, 1)
    y_lin = np.polyval(coeffs_lin, x)
    results.append(("linear", coeffs_lin, rmse(y, y_lin)))

    # --- Polynomial degree 2 ---
    coeffs_p2 = np.polyfit(x, y, 2)
    y_p2 = np.polyval(coeffs_p2, x)
    results.append(("polynomial (degree 2)", coeffs_p2, rmse(y, y_p2)))

    # --- Polynomial degree 3 ---
    coeffs_p3 = np.polyfit(x, y, 3)
    y_p3 = np.polyval(coeffs_p3, x)
    results.append(("polynomial (degree 3)", coeffs_p3, rmse(y, y_p3)))

    # --- Polynomial degree 4 ---
    coeffs_p4 = np.polyfit(x, y, 4)
    y_p4 = np.polyval(coeffs_p4, x)
    results.append(("polynomial (degree 4)", coeffs_p4, rmse(y, y_p4)))

    # --- Log-linear: log(thrust) = a*pwm + b -> thrust = exp(a*pwm + b) ---
    y_positive = np.maximum(y, 1e-6)
    log_y = np.log(y_positive)
    coeffs_log = np.polyfit(x, log_y, 1)
    y_log = np.exp(np.polyval(coeffs_log, x))
    results.append(("exponential (log-linear)", coeffs_log, rmse(y, y_log)))

    # Print errors and pick best
    print(f"PWM -> {target.capitalize()} regression (RMSE in {unit}):")
    best_name, best_coeffs, best_rmse = results[0]
    for name, coeffs, err in results:
        print(f"  {name}: RMSE = {err:.6f} {unit}")
        if err < best_rmse:
            best_rmse = err
            best_name = name
            best_coeffs = coeffs

    is_exp = "exponential" in best_name

    def predict(pwm_val: float | np.ndarray) -> float | np.ndarray:
        if is_exp:
            return np.exp(np.polyval(best_coeffs, np.asarray(pwm_val, dtype=float)))
        return np.polyval(best_coeffs, np.asarray(pwm_val, dtype=float))

    print(f"  -> Best model: {best_name} (RMSE = {best_rmse:.6f} {unit})")

    # Save best parameters for inference via pwm_to_thrust(..., target=...)
    out_path = save_path if save_path is not None else _default_params_path(target)
    params = {
        "target": target,
        "unit": unit,
        "model_name": best_name,
        "coeffs": best_coeffs.tolist(),
        "is_exponential": is_exp,
        "rmse": best_rmse,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"  -> Saved model to {out_path}")

    return predict, best_name, best_rmse


if __name__ == "__main__":
    # Thrust model
    # predict_fn, name, rmse_val = PWM2Thrust_regression(target="thrust")
    # print(f"Example: PWM 1500 -> Thrust = {predict_fn(1500):.2f} N")
    print(f"pwm_to_thrust(1500) = {pwm_to_thrust(1500, target='thrust'):.2f} N")
    
    
    # Power (energy) model
    # predict_power, name_p, rmse_p = PWM2Thrust_regression(target="power")
    # print(f"Example: PWM 1500 -> Power = {predict_power(1500):.2f} W")
    print(f"pwm_to_thrust(1500, target='power') = {pwm_to_thrust(1500, target='power'):.2f} W")