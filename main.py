from autocipher.ae_training import run_ae_training
from autocipher.clf_training import run_clf_training
from autocipher.parsing import parse_train_args


if __name__ == '__main__':
    args = parse_train_args()
    run_ae_training(args)
    # run_clf_training(args)
