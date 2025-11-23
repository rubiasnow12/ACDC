import collections
import pickle
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def get_patient_pathology(info_file_path: Path) -> str:
    """Reads the 'Group' information from an ACDC Info.cfg file."""
    with open(info_file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('Group:'):
                # 分割 "Group: DCM" 并获取 "DCM"
                return line.split(':')[1].strip()
    return "Unknown" # 如果找不到 Group 信息，返回默认值

def create_splits(config_path: str):
    """
    Scans the ACDC dataset, performs a stratified split by pathology,
    and saves the patient ID splits to a pickle file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['seed']
    data_root = Path(config['data_root'])
    splits_file_path = Path(config['splits_file'])
    train_ratio = config['split_ratios']['train']
    val_ratio = config['split_ratios']['val']

    # 确保输出目录存在
    splits_file_path.parent.mkdir(exist_ok=True)

    # 1. 扫描所有患者并读取他们的病理信息
    patients_by_pathology = collections.defaultdict(list)
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        info_cfg_path = patient_dir / "Info.cfg"
        if info_cfg_path.exists():
            # 使用新的辅助函数读取病理信息
            pathology = get_patient_pathology(info_cfg_path)
            patients_by_pathology[pathology].append(patient_dir.name)
        else:
            print(f"Warning: Could not find Info.cfg for {patient_dir.name}")

    print("Patients per pathology group:")
    for group, patients in patients_by_pathology.items():
        print(f"- {group}: {len(patients)} patients")

    # 2. 执行分层抽样
    train_patients, val_patients, test_patients = [], [], []
    
    rng = np.random.default_rng(seed)

    for group, patients in patients_by_pathology.items():
        rng.shuffle(patients)
        
        group_train, group_temp = train_test_split(
            patients, train_size=train_ratio, random_state=seed
        )
        
        if not group_temp: # 如果临时集为空，则跳过
            train_patients.extend(group_train)
            continue

        relative_val_ratio = val_ratio / (1 - train_ratio)
        
        # 确保当只有一个样本时不会出错
        if len(group_temp) == 1:
            group_val = group_temp
            group_test = []
        else:
            group_val, group_test = train_test_split(
                group_temp, train_size=relative_val_ratio, random_state=seed
            )
        
        train_patients.extend(group_train)
        val_patients.extend(group_val)
        test_patients.extend(group_test)

    splits = {
        "train": sorted(train_patients),
        "val": sorted(val_patients),
        "test": sorted(test_patients),
    }

    # 3. 保存划分结果
    with open(splits_file_path, "wb") as f:
        pickle.dump(splits, f)

    print("\nData split completed successfully!")
    print(f"Train patients: {len(splits['train'])}")
    print(f"Validation patients: {len(splits['val'])}")
    print(f"Test patients: {len(splits['test'])}")
    print(f"Splits saved to: {splits_file_path}")


if __name__ == "__main__":
    create_splits("configs/config.yaml")