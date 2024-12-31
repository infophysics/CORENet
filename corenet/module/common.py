"""
Module data
"""

module_types = {
    "ml":   [
        "training", "contrastive_training",
        "inference",
        "hyper_parameter_scan", "contrastive_hyper_parameter_scan",
        "linear_evaluation",
        "model_analyzer"
    ],
}

module_aliases = {
    "ml":               "MachineLearningModule",
    "machine_learning": "MachineLearningModule",
    "machinelearning":  "MachineLearningModule",
    "MachineLearning":  "MachineLearningModule",
}
