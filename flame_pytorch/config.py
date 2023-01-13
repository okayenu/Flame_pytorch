import argparse

parser = argparse.ArgumentParser(description="FLAME model")

parser.add_argument(
    "--flame_model_path",
    type=str,
    default="./model/generic_model.pkl",
    help="flame model path",
)
