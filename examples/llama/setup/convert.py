import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="root directory", default="llama-7b-hf")
    args = parser.parse_args()

    for f in os.listdir(args.src):
        if not f.endswith(".bin"):
            continue
        print(f"Loading {f}")
        sd = torch.load(os.path.join(args.src, f))
        for key, tensor in sd.items():
            print("Saving", key, tensor.shape, tensor.dtype)
            path = os.path.sep.join(key.split("."))
            os.makedirs(os.path.join(args.src, os.path.dirname(path)), exist_ok=True)
            np_array = tensor.numpy()
            with open(os.path.join(args.src, path), "w") as fp:
                np_array.tofile(fp)
            del np_array
        del sd


if __name__ == "__main__":
    main()