# tools/export_intro_motivation_cases.py

import os
import json
import argparse

import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from core.models.builder import build_model
from core.dataset.transforms import build_transform


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True
    )

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--out_dir",
        default="./results/intro_motivation"
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--max_scan",
        type=int,
        default=5000
    )

    return parser.parse_args()



class MotivationDataset(Dataset):

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

        self.transform = build_transform(
            train=False
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = (
            row["path"]
            if "path" in self.df.columns
            else row["image"]
        )

        label = int(row["label"])

        img = Image.open(
            path
        ).convert("RGB")


        img_tensor = self.transform(
            img
        )


        return {
            "image": img_tensor,
            "path": path,
            "label": label
        }



def load_model(
        config,
        checkpoint,
        device
):

    print("Loading model...")

    model = build_model(
        config
    )


    ckpt = torch.load(
        checkpoint,
        map_location="cpu"
    )


    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]


    new_state = {}

    for k, v in ckpt.items():

        if k.startswith("module."):
            k = k[7:]

        new_state[k] = v


    missing, unexpected = model.load_state_dict(
        new_state,
        strict=False
    )


    print(
        f"Checkpoint loaded."
    )

    print(
        f"Missing keys: {len(missing)}"
    )

    print(
        f"Unexpected keys: {len(unexpected)}"
    )


    model.to(device)

    model.eval()

    return model



def fake_prob(logits):

    if logits.ndim == 2:

        return torch.softmax(
            logits,
            dim=1
        )[:,1]


    return torch.sigmoid(
        logits.reshape(-1)
    )



def extract_outputs(
        outputs
):

    spatial = fake_prob(
        outputs["spatial_logits"]
    )


    recon = fake_prob(
        outputs["fire_logits"]
    )


    final = fake_prob(
        outputs["logits"]
    )


    router = outputs.get(
        "router_weights",
        None
    )


    if router is not None:

        spatial_w = router[:,0]

        recon_w = router[:,1]

    else:

        spatial_w = torch.zeros_like(final)

        recon_w = torch.zeros_like(final)


    return (
        spatial,
        recon,
        final,
        spatial_w,
        recon_w
    )



def save_case(
        item,
        save_path,
        title
):

    img = Image.open(
        item["path"]
    ).convert("RGB")


    plt.figure(
        figsize=(5,7)
    )


    plt.imshow(
        img
    )

    plt.axis(
        "off"
    )


    text = (
        f"{title}\n\n"

        f"Spatial:\n"
        f"{item['spatial_prob']:.3f} "
        f"{'Fake' if item['spatial_pred'] else 'Real'}\n\n"

        f"Reconstruction:\n"
        f"{item['recon_prob']:.3f} "
        f"{'Fake' if item['recon_pred'] else 'Real'}\n\n"

        f"Routing:\n"
        f"Spatial {item['router_spatial']:.3f}\n"
        f"Recon {item['router_recon']:.3f}\n\n"

        f"Fusion:\n"
        f"{item['final_prob']:.3f} "
        f"{'Fake' if item['final_pred'] else 'Real'}"
    )


    plt.figtext(
        0.5,
        0.02,
        text,
        ha="center",
        fontsize=9
    )


    plt.tight_layout()


    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()



def main():

    args = get_args()


    os.makedirs(
        args.out_dir,
        exist_ok=True
    )


    device = args.device


    model = load_model(
        args.config,
        args.checkpoint,
        device
    )


    print("Loading dataset...")


    dataset = MotivationDataset(
        args.csv
    )


    print(
        f"Dataset size: {len(dataset)}"
    )


    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    case_a = None
    case_b = None


    scanned = 0


    print("Start searching cases...")


    with torch.no_grad():

        for batch in tqdm(
            loader,
            desc="Inference"
        ):


            if scanned >= args.max_scan:
                break


            images = batch["image"].to(
                device
            )


            outputs = model(
                images,
                return_aux=True
            )


            spatial, recon, final, sw, rw = extract_outputs(
                outputs
            )


            paths = batch["path"]

            labels = batch["label"]


            for i in range(
                len(paths)
            ):


                if labels[i].item() != 1:
                    continue


                item = {

                    "path": paths[i],

                    "spatial_prob": float(spatial[i]),

                    "recon_prob": float(recon[i]),

                    "final_prob": float(final[i]),

                    "spatial_pred": int(spatial[i] >= 0.5),

                    "recon_pred": int(recon[i] >= 0.5),

                    "final_pred": int(final[i] >= 0.5),

                    "router_spatial": float(sw[i]),

                    "router_recon": float(rw[i])

                }



                if (
                    case_a is None
                    and item["spatial_pred"] == 0
                    and item["recon_pred"] == 1
                    and item["final_pred"] == 1
                ):

                    case_a = item

                    print(
                        "\nFound Case A"
                    )



                if (
                    case_b is None
                    and item["spatial_pred"] == 1
                    and item["recon_pred"] == 0
                    and item["final_pred"] == 1
                ):

                    case_b = item

                    print(
                        "\nFound Case B"
                    )


                if case_a and case_b:

                    break


            scanned += len(paths)


            if case_a and case_b:
                break



    result = {

        "case_A": case_a,

        "case_B": case_b

    }


    with open(
        os.path.join(
            args.out_dir,
            "selected_cases.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )


    if case_a:

        save_case(
            case_a,
            os.path.join(
                args.out_dir,
                "case_A.png"
            ),
            "Case A: Spatial Failure"
        )


    if case_b:

        save_case(
            case_b,
            os.path.join(
                args.out_dir,
                "case_B.png"
            ),
            "Case B: Reconstruction Failure"
        )


    print("\nFinished.")
    print(
        f"Saved to: {args.out_dir}"
    )



if __name__ == "__main__":

    main()