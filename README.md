# ActiveBayesify

## Setup

* In the `config.ini` there must be set generic configuration parameters for the system (e.g. paths).
* The system was tested with Python 3.9.2 as well as 3.10.1
* There is a Dockerfile available to set up the system.
* Otherwise, necessary packages can be installed with `pip install -r requirements.txt`.
* Information about the different cli arguments can be found with `python3.9 pipeline.py --help`.

### Dockerfile
* Build container with `docker build -t activebayesify .`
* Run container with `docker run -it activebayesify bash`

## Experiments
* Some examples of experiments and their evaluations are listed in the table below:

| # Experiment     |                                                                           Command                                                                            |
|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Exp.1 for x264   |   `python3.9 pipeline.py --system-name "x264" --file "x264_ml_cfg_summed_only_top_functions" --output "random_learning_on_system_level" --random --system`   |
| Exp.2 for x264   |   `python3.9 pipeline.py --system-name "x264" --file "x264_ml_cfg_summed_only_top_functions" --output "active_learning_on_system_level" --active --system`   |
| Exp.3 for x264   |              `python3.9 pipeline.py --system-name "x264" --file "x264_ml_cfg" --output "random_learning_on_function_level" --random --function`              |
| Exp.1 for brotli | `python3.9 pipeline.py --system-name "brotli" --file "brotli_ml_cfg_summed_only_top_functions" --output "random_learning_on_system_level" --random --system` |
| Exp.2 for brotli | `python3.9 pipeline.py --system-name "brotli" --file "brotli_ml_cfg_summed_only_top_functions" --output "active_learning_on_system_level" --active --system` |
| Exp.3 for brotli |            `python3.9 pipeline.py --system-name "brotli" --file "brotli_ml_cfg" --output "random_learning_on_function_level" --random --function`            |
| Exp.1 for lrzip  |  `python3.9 pipeline.py --system-name "lrzip" --file "lrzip_ml_cfg_summed_only_top_functions" --output "random_learning_on_system_level" --random --system`  |
| Exp.2 for lrzip  |  `python3.9 pipeline.py --system-name "lrzip" --file "lrzip_ml_cfg_summed_only_top_functions" --output "active_learning_on_system_level" --active --system`  |
| Exp.3 for lrzip  |             `python3.9 pipeline.py --system-name "lrzip" --file "lrzip_ml_cfg" --output "random_learning_on_function_level" --random --function`             |

## Evaluations

| # Evaluation     |                                                                      Command                                                                      |
|:-----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------:|
| Exp.1 for x264   |    `python3.9 evaluation.py --system-name "x264" --base-file "random_learning_on_system_level" --base-directory system_level/random_learning/`    |
| Exp.2 for x264   |    `python3.9 evaluation.py --system-name "x264" --base-file "active_learning_on_system_level" --base-directory system_level/active_learning/`    |
| Exp.3 for x264   |  `python3.9 evaluation.py --system-name "x264" --base-file "random_learning_on_function_level" --base-directory function_level/random_learning/`  |
| Exp.1 for brotli |   `python3.9 evaluation.py --system-name "brotli" --base-file "random_learning_on_system_level" --base-directory system_level/random_learning/`   |
| Exp.2 for brotli |   `python3.9 evaluation.py --system-name "brotli" --base-file "active_learning_on_system_level" --base-directory system_level/active_learning/`   |
| Exp.3 for brotli | `python3.9 evaluation.py --system-name "brotli" --base-file "random_learning_on_function_level" --base-directory function_level/random_learning/` |
| Exp.1 for lrzip  |   `python3.9 evaluation.py --system-name "lrzip" --base-file "random_learning_on_system_level" --base-directory system_level/random_learning/`    |
| Exp.2 for lrzip  |   `python3.9 evaluation.py --system-name "lrzip" --base-file "active_learning_on_system_level" --base-directory system_level/active_learning/`    |
| Exp.3 for lrzip  | `python3.9 evaluation.py --system-name "lrzip" --base-file "random_learning_on_function_level" --base-directory function_level/random_learning/`  | 

## Logs
* Logs can be found in the dedicated system directory at `./final/logs/`

## Results
* results can be found in the dedicated system directory at `./final/results/SYSTEM_NAME_HERE/`
* images from the evaluation can be found in the dedicated system directory at `./final/images/SYSTEM_NAME_HERE/` 

## Reproducibility

Reproducibility is connected to the repetition number. Each repetition has its own random state.
This enables comparability between the different experiments.

## Tests

Unit tests can be found in the tests directory.
