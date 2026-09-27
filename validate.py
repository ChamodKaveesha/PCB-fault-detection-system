from ultralytics import YOLO

MODEL_PATH = "CK.pt"          
DATA_YAML = "data.yaml"       


def run_split(model, split_name):
    print(f"\n\n===== Running on '{split_name}' split =====")
    metrics = model.val(data=DATA_YAML, split=split_name)

    print(f"\n----- {split_name.upper()} RESULTS -----")
    print(f"Precision (mp):      {metrics.box.mp:.4f}")
    print(f"Recall (mr):         {metrics.box.mr:.4f}")
    print(f"mAP50:               {metrics.box.map50:.4f}")
    print(f"mAP50-95:            {metrics.box.map:.4f}")
    print("--------------------------------")
    print("Per-class results:")
    for i in range(len(metrics.box.ap50)):
        name = metrics.names.get(i, f"class_{i}")
        print(f"  {name}: mAP50 = {metrics.box.ap50[i]:.4f}")
    print(f"\nFull results, confusion matrix, and plots saved to: {metrics.save_dir}")


def main():
    model = YOLO(MODEL_PATH)

    run_split(model, "val")

    run_split(model, "test")


if __name__ == "__main__":
    main()
