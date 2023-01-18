import argparse

parser = argparse.ArgumentParser(description="FLAME model")

parser.add_argument(
    "--flame_model_path",
    type=str,
    default="./model/generic_model.pkl",
    help="flame model path",
)

parser.add_argument(
    "--static_landmark_embedding_path",
    type=str,
    default="./model/flame_static_embedding.pkl",
    help="Static landmark embeddings path for FLAME",
)

parser.add_argument(
    "--dynamic_landmark_embedding_path",
    type=str,
    default="./model/flame_dynamic_embedding.npy",
    help="Dynamic contour embedding path for FLAME",
)

# FLAME hyper-parameters

parser.add_argument(
    "--shape_params", type=int, default=100, help="the number of shape parameters"
)

parser.add_argument(
    "--expression_params",
    type=int,
    default=50,
    help="the number of expression parameters",
)

parser.add_argument(
    "--pose_params", type=int, default=6, help="the number of pose parameters"
