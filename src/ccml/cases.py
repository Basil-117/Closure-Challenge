"""Case registry and the suggested train / validation / test split from the
Closure Challenge README. TEST cases are only ever read for their RANS
(input) fields -- never their LES fields -- so nothing leaks into training.
"""
from pathlib import Path

TEST = [
    "alpha_15_13929_4048", "alpha_15_13929_2024",
    "alpha_05_4071_4048", "alpha_05_4071_2024",
    "AR_1_Ret_360", "AR_3_Ret_360", "AR_14_Ret_180",
    "NASA_2DWMH",
]
VAL = [
    "alpha_05_10071_4048", "alpha_05_10071_2024",
    "alpha_15_7929_4048", "alpha_15_7929_2024",
    "AR_7_Ret_180",
]
_DUCT_TRAIN = ["AR_1_Ret_180", "AR_3_Ret_180", "AR_5_Ret_180", "AR_10_Ret_180"]


def case_path(data_root: Path, name: str) -> Path:
    data_root = Path(data_root)
    if name.startswith("alpha_"):
        fam = "_".join(name.split("_")[:2])
        return data_root / "Parm_PH_29" / fam / name
    if name.startswith("AR_"):
        return data_root / "DUCT" / name
    return {"CBFS": data_root / "CBFS", "PH_Breuer": data_root / "PH_Breuer",
            "NASA_2DWMH": data_root / "NASA_2DWMH"}[name]


def all_hill_cases(data_root: Path) -> list[str]:
    root = Path(data_root) / "Parm_PH_29"
    return sorted(p.name for fam in root.iterdir() for p in fam.iterdir() if p.is_dir())


def split(data_root: Path) -> dict[str, list[str]]:
    hills_train = [c for c in all_hill_cases(data_root) if c not in TEST + VAL]
    assert len(hills_train) == 21, len(hills_train)
    train = hills_train + _DUCT_TRAIN + ["CBFS", "PH_Breuer"]
    assert not set(train) & set(TEST + VAL)
    return {"train": train, "val": VAL, "test": TEST}
