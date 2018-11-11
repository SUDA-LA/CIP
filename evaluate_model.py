import sys, os
import HMM, LinearModel, LogLinearModel, GlobalLinearModel, CRF

__console_out__ = sys.stdout

__result_path__ = './result'

if not os.path.exists(__result_path__):
    os.mkdir(__result_path__)