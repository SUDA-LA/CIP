import sys, os, datetime
import numpy as np
import HMM, LinearModel, LogLinearModel, GlobalLinearModel, CRF

__console_out__ = sys.stdout

__result_path__ = './result'

if not os.path.exists(__result_path__):
    os.mkdir(__result_path__)


def evaluate(tagger, result_path, config=None, bigdata=False):
    if bigdata:
        train_path = '.\\bigdata\\train.conll'
        dev_path = '.\\bigdata\\dev.conll'
        test_path = '.\\bigdata\\test.conll'
        suffix = '_big.res'
    else:
        train_path = '.\\data\\train.conll'
        dev_path = '.\\data\\dev.conll'
        test_path = None
        suffix = '.res'
    with open(__result_path__ + '/' + result_path + suffix, 'w') as result_file:
        start = datetime.datetime.now()
        print(f"{result_path} {'bigdata' if bigdata else 'data'} train start at {start}")
        sys.stdout = result_file
        tagger.train(train_path,
                     dev_path=dev_path,
                     test_path=test_path,
                     config=config)
        sys.stdout = __console_out__
        end = datetime.datetime.now()
        print(f"{result_path} {'bigdata' if bigdata else 'data'} train finish at {end} spend {end - start}s")


# HMM
_tagger = HMM.Tagger.Tagger()
_result_path = _tagger.model_name
_config = _tagger.Config(0.3, evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=False)
_config = _tagger.Config(0.01, evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=True)


# LinearModel
_tagger = LinearModel.Tagger.Tagger()
_optimizedTagger = LinearModel.OptimizedTagger.Tagger()

# Linear Model vanilla
_result_path = _tagger.model_name
_optimized_result_path = _optimizedTagger.model_name
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=False,
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Linear Model averaged perceptron
_result_path = _tagger.model_name + '_averaged_perceptron'
_optimized_result_path = _optimizedTagger.model_name + '_averaged_perceptron'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=True,
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Linear Model random
_result_path = _tagger.model_name + '_random'
_optimized_result_path = _optimizedTagger.model_name + '_random'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=False,
                         random_lr=lambda: 0.8 + 0.4 * np.random.random(),
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Linear Model random averaged perceptron
_result_path = _tagger.model_name + '_random_averaged_perceptron'
_optimized_result_path = _optimizedTagger.model_name + '_random_averaged_perceptron'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=True,
                         random_lr=lambda: 0.8 + 0.4 * np.random.random(),
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Log Linear Model
_tagger = LogLinearModel.Tagger.Tagger()
_optimizedTagger = LogLinearModel.OptimizedTagger.Tagger()

# Log Linear Model vanilla
_result_path = _tagger.model_name
_optimized_result_path = _optimizedTagger.model_name
_config = _tagger.Config(learning_rate=0.5,  # data 0.5 is fine
                         c=0,
                         rho=1,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
_config = _tagger.Config(learning_rate=0.1,  # data 0.5 is fine
                         c=0,
                         rho=1,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Log Linear Model Anneal
_result_path = _tagger.model_name + '_anneal'
_optimized_result_path = _optimizedTagger.model_name + '_anneal'
_config = _tagger.Config(learning_rate=0.5,  # data 0.5 is fine
                         c=0.0001,
                         rho=0.96,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
_config = _tagger.Config(learning_rate=0.1,  # data 0.5 is fine
                         c=0.0001,
                         rho=0.96,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Global Linear Model
_tagger = GlobalLinearModel.Tagger.Tagger()
_optimizedTagger = GlobalLinearModel.OptimizedTagger.Tagger()

# Global Linear Model vanilla
_result_path = _tagger.model_name
_optimized_result_path = _optimizedTagger.model_name
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=False,
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Global Linear Model averaged perceptron
_result_path = _tagger.model_name + '_averaged_perceptron'
_optimized_result_path = _optimizedTagger.model_name + '_averaged_perceptron'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=True,
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Global Linear Model random
_result_path = _tagger.model_name + '_random'
_optimized_result_path = _optimizedTagger.model_name + '_random'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=False,
                         random_lr=lambda: 0.8 + 0.4 * np.random.random(),
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# Global Linear Model random averaged perceptron
_result_path = _tagger.model_name + '_random_averaged_perceptron'
_optimized_result_path = _optimizedTagger.model_name + '_random_averaged_perceptron'
_config = _tagger.Config(stop_threshold=0,
                         max_iter=50,
                         averaged_perceptron=True,
                         random_lr=lambda: 0.8 + 0.4 * np.random.random(),
                         evaluate_mode=True
                         )
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# CRF
_tagger = CRF.Tagger.Tagger()
_optimizedTagger = CRF.OptimizedTagger.Tagger()

# CRF vanilla
_result_path = _tagger.model_name
_optimized_result_path = _optimizedTagger.model_name
_config = _tagger.Config(learning_rate=0.5,  # data 0.5 is fine
                         c=0,
                         rho=1,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
_config = _tagger.Config(learning_rate=0.1,  # data 0.5 is fine
                         c=0,
                         rho=1,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)

# CRF Anneal
_result_path = _tagger.model_name + '_anneal'
_optimized_result_path = _optimizedTagger.model_name + '_anneal'
_config = _tagger.Config(learning_rate=0.5,  # data 0.5 is fine
                         c=0.0001,
                         rho=0.96,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=False)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=False)
_config = _tagger.Config(learning_rate=0.1,  # data 0.5 is fine
                         c=0.0001,
                         rho=0.96,
                         delay_step=100000,
                         max_iter=80,
                         batch_size=50,
                         evaluate_mode=True)
evaluate(_tagger, _result_path, _config, bigdata=True)
evaluate(_optimizedTagger, _optimized_result_path, _config, bigdata=True)
