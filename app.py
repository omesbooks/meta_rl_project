"""
PyCaret Trainer GUI
-------------------
แนบไฟล์ dataset (CSV/XLSX) -> เลือก target column + task -> เทรน -> ได้ไฟล์ .pkl

รองรับ Time-Series Mode:
  - Time-based split (ไม่ shuffle)
  - fold_strategy='timeseries' (prevent leakage)
  - Walk-Forward Validation (optional)
"""

import os
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import pandas as pd


TASKS = {
    "Classification": "pycaret.classification",
    "Regression": "pycaret.regression",
    "Clustering": "pycaret.clustering",
    "Anomaly Detection": "pycaret.anomaly",
    "Time Series": "pycaret.time_series",
}


class TrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyCaret Trainer")
        self.geometry("820x780")

        self.df: pd.DataFrame | None = None
        self.dataset_path: str | None = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # --- Row 1: file picker ---
        frm_file = ttk.LabelFrame(self, text="1) Dataset")
        frm_file.pack(fill="x", **pad)

        self.lbl_file = ttk.Label(frm_file, text="ยังไม่ได้เลือกไฟล์", foreground="gray")
        self.lbl_file.pack(side="left", padx=8, pady=8)
        ttk.Button(frm_file, text="เลือกไฟล์ CSV/XLSX...", command=self.pick_file).pack(
            side="right", padx=8, pady=8
        )

        # --- Row 2: task + target ---
        frm_cfg = ttk.LabelFrame(self, text="2) การตั้งค่าการเทรน")
        frm_cfg.pack(fill="x", **pad)

        ttk.Label(frm_cfg, text="ประเภทงาน:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.cbo_task = ttk.Combobox(frm_cfg, values=list(TASKS.keys()), state="readonly", width=25)
        self.cbo_task.current(0)
        self.cbo_task.grid(row=0, column=1, padx=8, pady=6, sticky="w")
        self.cbo_task.bind("<<ComboboxSelected>>", self._on_task_change)

        ttk.Label(frm_cfg, text="คอลัมน์ target:").grid(row=0, column=2, sticky="w", padx=8, pady=6)
        self.cbo_target = ttk.Combobox(frm_cfg, state="readonly", width=25)
        self.cbo_target.grid(row=0, column=3, padx=8, pady=6, sticky="w")

        ttk.Label(frm_cfg, text="ชื่อไฟล์ model:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.ent_name = ttk.Entry(frm_cfg, width=30)
        self.ent_name.insert(0, "my_model")
        self.ent_name.grid(row=1, column=1, padx=8, pady=6, sticky="w")

        ttk.Label(frm_cfg, text="โฟลเดอร์ output:").grid(row=1, column=2, sticky="w", padx=8, pady=6)
        self.ent_out = ttk.Entry(frm_cfg, width=28)
        self.ent_out.insert(0, str(Path.cwd()))
        self.ent_out.grid(row=1, column=3, padx=8, pady=6, sticky="w")
        ttk.Button(frm_cfg, text="...", width=3, command=self.pick_out_dir).grid(
            row=1, column=4, padx=4
        )

        # --- Row 3: Time-Series options ---
        frm_ts = ttk.LabelFrame(self, text="3) Time-Series Options (สำหรับข้อมูลเทรดดิ้ง)")
        frm_ts.pack(fill="x", **pad)

        self.var_ts_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_ts,
            text="เปิด Time-Series Mode (ไม่ shuffle, ใช้ timeseries fold) — แนะนำสำหรับข้อมูลราคา",
            variable=self.var_ts_mode,
            command=self._on_ts_toggle,
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=4)

        ttk.Label(frm_ts, text="คอลัมน์เวลา (สำหรับเรียงลำดับ):").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.cbo_time = ttk.Combobox(frm_ts, state="disabled", width=22)
        self.cbo_time.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        ttk.Label(frm_ts, text="Test size (%):").grid(row=1, column=2, sticky="w", padx=8, pady=4)
        self.ent_test = ttk.Entry(frm_ts, width=8)
        self.ent_test.insert(0, "20")
        self.ent_test.grid(row=1, column=3, padx=8, pady=4, sticky="w")

        ttk.Label(frm_ts, text="CV folds:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.ent_folds = ttk.Entry(frm_ts, width=8)
        self.ent_folds.insert(0, "5")
        self.ent_folds.grid(row=2, column=1, padx=8, pady=4, sticky="w")

        self.var_imbalance = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_ts, text="Fix class imbalance (SMOTE)", variable=self.var_imbalance
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=8, pady=4)

        self.var_walkforward = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_ts,
            text="เพิ่ม Walk-Forward Validation (หลังเทรนหลัก) — ช้าลงแต่ robust กว่า",
            variable=self.var_walkforward,
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=4)

        ttk.Label(frm_ts, text="WF windows:").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        self.ent_wf_windows = ttk.Entry(frm_ts, width=8)
        self.ent_wf_windows.insert(0, "5")
        self.ent_wf_windows.grid(row=4, column=1, padx=8, pady=4, sticky="w")

        # --- Row 4: train button ---
        frm_run = ttk.Frame(self)
        frm_run.pack(fill="x", **pad)
        self.btn_train = ttk.Button(frm_run, text="เริ่มเทรนโมเดล", command=self.start_training)
        self.btn_train.pack(side="left", padx=8)
        self.progress = ttk.Progressbar(frm_run, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=8)

        # --- Row 5: log ---
        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill="both", expand=True, **pad)
        self.txt_log = scrolledtext.ScrolledText(frm_log, height=18, font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True, padx=4, pady=4)

        self.log("พร้อมใช้งาน. เลือกไฟล์ dataset เพื่อเริ่มต้น.")

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def log(self, msg: str):
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.update_idletasks()

    def pick_file(self):
        path = filedialog.askopenfilename(
            title="เลือกไฟล์ dataset",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("อ่านไฟล์ไม่ได้", str(e))
            return

        self.df = df
        self.dataset_path = path
        self.lbl_file.config(
            text=f"{os.path.basename(path)}  ({len(df):,} rows x {len(df.columns)} cols)",
            foreground="black",
        )
        self.cbo_target["values"] = list(df.columns)
        self.cbo_time["values"] = list(df.columns)
        if len(df.columns):
            self.cbo_target.current(len(df.columns) - 1)
            # auto-detect time column
            for c in df.columns:
                if any(k in c.lower() for k in ("time", "date")):
                    self.cbo_time.set(c)
                    break
        self.log(f"โหลดไฟล์: {path}")
        self.log(f"คอลัมน์: {list(df.columns)}")

        # show class balance if looks like classification
        if len(df.columns):
            last_col = df.columns[-1]
            if df[last_col].dtype == "object" or df[last_col].nunique() < 10:
                try:
                    vc = df[last_col].value_counts(normalize=True)
                    self.log(f"Class balance ของ '{last_col}':")
                    for k, v in vc.items():
                        self.log(f"  {k}: {v:.1%}")
                except Exception:
                    pass

    def pick_out_dir(self):
        d = filedialog.askdirectory(title="เลือกโฟลเดอร์ output")
        if d:
            self.ent_out.delete(0, "end")
            self.ent_out.insert(0, d)

    def _on_task_change(self, _evt=None):
        task = self.cbo_task.get()
        needs_target = task not in ("Clustering", "Anomaly Detection")
        self.cbo_target.configure(state="readonly" if needs_target else "disabled")

    def _on_ts_toggle(self):
        state = "readonly" if self.var_ts_mode.get() else "disabled"
        self.cbo_time.configure(state=state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def start_training(self):
        if self.df is None:
            messagebox.showwarning("ยังไม่มีข้อมูล", "กรุณาเลือกไฟล์ dataset ก่อน")
            return
        task = self.cbo_task.get()
        target = self.cbo_target.get() if self.cbo_target["state"] != "disabled" else None
        name = self.ent_name.get().strip() or "my_model"
        out_dir = self.ent_out.get().strip() or str(Path.cwd())

        if task in ("Classification", "Regression", "Time Series") and not target:
            messagebox.showwarning("ไม่ได้เลือก target", "กรุณาเลือกคอลัมน์ target")
            return

        ts_opts = {
            "enabled": self.var_ts_mode.get(),
            "time_col": self.cbo_time.get() if self.var_ts_mode.get() else None,
            "test_pct": float(self.ent_test.get() or 20) / 100.0,
            "folds": int(self.ent_folds.get() or 5),
            "imbalance": self.var_imbalance.get(),
            "walkforward": self.var_walkforward.get(),
            "wf_windows": int(self.ent_wf_windows.get() or 5),
        }

        self.btn_train.configure(state="disabled")
        self.progress.start(10)
        t = threading.Thread(
            target=self._train_worker,
            args=(task, target, name, out_dir, ts_opts),
            daemon=True,
        )
        t.start()

    def _train_worker(self, task, target, name, out_dir, ts):
        try:
            self.log(f"\n=== เริ่มเทรน: {task} ===")
            module_name = TASKS[task]
            self.log(f"import {module_name} ...")
            import importlib

            mod = importlib.import_module(module_name)

            df = self.df.copy()

            # sort by time if TS mode
            if ts["enabled"] and ts["time_col"]:
                self.log(f"เรียงข้อมูลตาม: {ts['time_col']}")
                try:
                    df[ts["time_col"]] = pd.to_datetime(df[ts["time_col"]])
                except Exception:
                    pass
                df = df.sort_values(ts["time_col"]).reset_index(drop=True)
                # drop time col (ไม่ให้เป็น feature)
                df_train = df.drop(columns=[ts["time_col"]])
            else:
                df_train = df

            # 🚨 Leakage guard — drop คอลัมน์ที่รู้ว่าเป็น "อนาคต"
            leaky = [c for c in df_train.columns
                     if c != target and any(k in c.lower() for k in
                        ("future_", "future", "forward_", "label_raw", "next_"))]
            if leaky:
                self.log(f"⚠ drop leaky columns (future info): {leaky}")
                df_train = df_train.drop(columns=leaky)

            # drop non-numeric identifiers ที่ไม่ควรใช้เป็น feature
            drop_ids = [c for c in df_train.columns
                        if c != target and c.lower() in ("symbol", "ticker", "id")]
            if drop_ids:
                self.log(f"  drop identifier columns: {drop_ids}")
                df_train = df_train.drop(columns=drop_ids)

            # build setup kwargs
            setup_kwargs = dict(data=df_train, session_id=123, verbose=False)
            if task in ("Classification", "Regression"):
                setup_kwargs["target"] = target
                if ts["enabled"]:
                    setup_kwargs["data_split_shuffle"] = False
                    setup_kwargs["data_split_stratify"] = False
                    setup_kwargs["fold_strategy"] = "timeseries"
                    setup_kwargs["fold"] = ts["folds"]
                    setup_kwargs["train_size"] = 1.0 - ts["test_pct"]
                    self.log(f"  Time-based split: train {(1-ts['test_pct'])*100:.0f}% / test {ts['test_pct']*100:.0f}%")
                    self.log(f"  Fold strategy: timeseries ({ts['folds']} folds)")
                if ts["imbalance"] and task == "Classification":
                    setup_kwargs["fix_imbalance"] = True
                    self.log("  Fix imbalance: ON (SMOTE)")
            elif task == "Time Series":
                setup_kwargs["target"] = target
                setup_kwargs["fh"] = 10
                setup_kwargs["fold"] = ts["folds"]

            self.log("setup() ...")
            if task in ("Clustering", "Anomaly Detection"):
                setup_kwargs.pop("target", None)
                mod.setup(**setup_kwargs)
            else:
                mod.setup(**setup_kwargs)

            # train
            if task in ("Clustering", "Anomaly Detection"):
                model_id = "kmeans" if task == "Clustering" else "iforest"
                self.log(f"create_model('{model_id}') ...")
                best = mod.create_model(model_id)
            else:
                self.log("compare_models() — อาจใช้เวลาสักครู่ ...")
                best = mod.compare_models()

            self.log(f"โมเดลที่ดีที่สุด: {type(best).__name__}")

            # ระบุ PyCaret model ID ของ best (เอาไว้ใช้ใน Walk-Forward ให้ตรงกัน)
            best_model_id = self._pycaret_model_id(mod, best, task)
            if best_model_id:
                self.log(f"PyCaret model ID: '{best_model_id}'")
            else:
                self.log("⚠ ระบุ model ID ไม่ได้ (จะใช้ 'lr' เป็น fallback ใน WF)")

            # finalize + save ก่อน เพื่อไม่ให้ setup ของ WF เข้ามาทับ env
            if task in ("Classification", "Regression", "Time Series"):
                self.log("finalize_model() ...")
                final_model = mod.finalize_model(best)
            else:
                final_model = best

            out_path = Path(out_dir) / name
            self.log(f"save_model -> {out_path}.pkl")
            mod.save_model(final_model, str(out_path))

            # Walk-Forward validation (ทำหลัง save เพราะจะ overwrite PyCaret env)
            if ts["enabled"] and ts["walkforward"] and task in ("Classification", "Regression"):
                self._run_walkforward(mod, df_train, target, task, ts, best_model_id)

            self.log("\n✓ เสร็จสิ้น!")
            self.after(0, lambda: messagebox.showinfo("สำเร็จ", f"บันทึก model ที่:\n{out_path}.pkl"))
        except Exception as e:
            self.log("\n✗ เกิดข้อผิดพลาด:")
            self.log(traceback.format_exc())
            err = str(e)
            self.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.after(0, self._train_done)

    # ------------------------------------------------------------------
    # Model ID lookup
    # ------------------------------------------------------------------
    # mapping จาก sklearn class name -> PyCaret model ID
    _MODEL_ID_MAP = {
        # Classification
        "LogisticRegression": "lr",
        "KNeighborsClassifier": "knn",
        "GaussianNB": "nb",
        "DecisionTreeClassifier": "dt",
        "SGDClassifier": "svm",
        "SVC": "rbfsvm",
        "GaussianProcessClassifier": "gpc",
        "MLPClassifier": "mlp",
        "RidgeClassifier": "ridge",
        "RandomForestClassifier": "rf",
        "QuadraticDiscriminantAnalysis": "qda",
        "AdaBoostClassifier": "ada",
        "GradientBoostingClassifier": "gbc",
        "LinearDiscriminantAnalysis": "lda",
        "ExtraTreesClassifier": "et",
        "XGBClassifier": "xgboost",
        "LGBMClassifier": "lightgbm",
        "CatBoostClassifier": "catboost",
        "DummyClassifier": "dummy",
        # Regression
        "LinearRegression": "lr",
        "Lasso": "lasso",
        "Ridge": "ridge",
        "ElasticNet": "en",
        "Lars": "lar",
        "LassoLars": "llar",
        "OrthogonalMatchingPursuit": "omp",
        "BayesianRidge": "br",
        "ARDRegression": "ard",
        "PassiveAggressiveRegressor": "par",
        "RANSACRegressor": "ransac",
        "TheilSenRegressor": "tr",
        "HuberRegressor": "huber",
        "KernelRidge": "kr",
        "SVR": "svm",
        "KNeighborsRegressor": "knn",
        "DecisionTreeRegressor": "dt",
        "RandomForestRegressor": "rf",
        "ExtraTreesRegressor": "et",
        "AdaBoostRegressor": "ada",
        "GradientBoostingRegressor": "gbr",
        "MLPRegressor": "mlp",
        "XGBRegressor": "xgboost",
        "LGBMRegressor": "lightgbm",
        "CatBoostRegressor": "catboost",
        "DummyRegressor": "dummy",
    }

    def _pycaret_model_id(self, mod, estimator, task):
        """หา PyCaret ID จาก sklearn estimator ของ best model."""
        # unwrap pipeline/wrapper ถ้ามี
        inner = estimator
        for attr in ("steps", "_final_estimator", "estimator", "regressor_", "classifier_"):
            if hasattr(inner, attr):
                val = getattr(inner, attr)
                if attr == "steps" and val:
                    inner = val[-1][1]
                elif attr != "steps":
                    inner = val
                break

        class_name = type(inner).__name__
        if class_name in self._MODEL_ID_MAP:
            return self._MODEL_ID_MAP[class_name]

        # fallback: ลองดูจาก mod.models() ถ้า available
        try:
            df_models = mod.models()
            name = type(inner).__name__.lower()
            for idx in df_models.index:
                row_name = str(df_models.loc[idx, "Name"]).lower().replace(" ", "")
                if name.replace("classifier", "").replace("regressor", "") in row_name:
                    return idx
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Walk-Forward Validation
    # ------------------------------------------------------------------
    def _run_walkforward(self, mod, df, target, task, ts, model_id):
        """
        แบ่งข้อมูลเป็น N windows แล้วเทรน-เทสต์ไล่ตามเวลา
        ใช้ model_id เดียวกับ best model (ไม่ fallback เป็น 'lr' แล้ว)
        """
        self.log("\n--- Walk-Forward Validation ---")

        if not model_id:
            model_id = "lr"
            self.log(f"  (ใช้ fallback '{model_id}' เพราะระบุ model ID ไม่ได้)")
        self.log(f"  ใช้โมเดล: '{model_id}' (ตัวเดียวกับ best)")

        n = len(df)
        n_windows = ts["wf_windows"]
        # แบ่งเป็น expanding window: train เริ่มจาก 1 ส่วน -> เพิ่มขึ้นเรื่อยๆ / test = 1 ส่วน
        window_size = n // (n_windows + 1)
        if window_size < 50:
            self.log(f"  ⚠ ข้อมูลน้อย ({n} แถว) สำหรับ {n_windows} windows — ข้าม WF")
            return

        import statistics
        from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score

        scores = []
        pred_col = "prediction_label"

        for i in range(n_windows):
            train_end = window_size * (i + 1)
            test_end = min(train_end + window_size, n)
            if test_end - train_end < 10:
                break

            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            self.log(
                f"  Window {i+1}/{n_windows}: "
                f"train[0:{train_end}] ({train_end} rows) -> "
                f"test[{train_end}:{test_end}] ({test_end - train_end} rows)"
            )

            try:
                # setup ใหม่ด้วย train window (ใช้ almost-all เป็น train, ไม่ต้องแบ่ง test อีก)
                setup_kw = dict(
                    data=train_df,
                    target=target,
                    session_id=123,
                    verbose=False,
                    data_split_shuffle=False,
                    data_split_stratify=False,
                    fold_strategy="timeseries",
                    fold=3,
                    train_size=0.95,
                )
                if ts.get("imbalance") and task == "Classification":
                    setup_kw["fix_imbalance"] = True

                mod.setup(**setup_kw)

                # สร้าง model ชนิดเดียวกับ best
                m = mod.create_model(model_id, verbose=False)

                # ทำนายบน test window
                pred = mod.predict_model(m, data=test_df, verbose=False)

                if task == "Classification":
                    y_true = test_df[target]
                    y_pred = pred[pred_col] if pred_col in pred.columns else pred["Label"]
                    acc = accuracy_score(y_true, y_pred)
                    try:
                        f1 = f1_score(y_true, y_pred, average="weighted")
                        self.log(f"    Accuracy: {acc:.4f} | F1(weighted): {f1:.4f}")
                    except Exception:
                        self.log(f"    Accuracy: {acc:.4f}")
                    scores.append(acc)
                else:
                    y_true = test_df[target]
                    y_pred = pred[pred_col] if pred_col in pred.columns else pred["Label"]
                    mae = mean_absolute_error(y_true, y_pred)
                    self.log(f"    MAE: {mae:.6f}")
                    scores.append(mae)
            except Exception as e:
                self.log(f"    (window error: {e})")

        if scores:
            mean_s = statistics.mean(scores)
            std_s = statistics.stdev(scores) if len(scores) > 1 else 0.0
            metric = "Accuracy" if task == "Classification" else "MAE"
            self.log(f"\n  === WF Summary ({metric}) ===")
            self.log(f"  mean: {mean_s:.4f} | std: {std_s:.4f} | windows: {len(scores)}")
            # robust check
            if task == "Classification" and len(scores) > 1:
                if std_s / max(mean_s, 1e-9) > 0.15:
                    self.log("  ⚠ std/mean สูง (>15%) — ผลแปรปรวนมาก อาจ overfit บางช่วง")
                else:
                    self.log("  ✓ ผลค่อนข้างเสถียรทั้ง window")

    def _train_done(self):
        self.progress.stop()
        self.btn_train.configure(state="normal")


if __name__ == "__main__":
    app = TrainerApp()
    app.mainloop()
