from py_experimenter.experimenter import PyExperimenter

experimenter = PyExperimenter(
    experiment_configuration_file_path="config.yaml",
    use_codecarbon=False
)
experimenter.fill_table_from_config()