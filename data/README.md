# Data

Datasets are intentionally **not redistributed** in this repository.

## BeliN

The scripts can load the public [BeliN dataset](https://github.com/akabircs/BeliN) directly from Hugging Face using the dataset identifier configured in `config/experiment_settings.yaml` (`osama39/BNHG`). You may also pass a local CSV.

## Shironaam

Obtain the [Shironaam dataset](https://github.com/dialect-ai/BenHeadGen) and place the required split locally, for example:

- `data/shironaam_train.csv` for the semantic-alignment analysis.
- `data/shironaam_test.csv` for the second-benchmark headline-generation experiment.

The scripts accept configurable column names and do not require the data to be stored in the repository.
