import sys
sys.path.append("pytracking")
sys.path.append("pytracking/pytracking")
sys.path.append("pytracking/ltr")

from pytracking.evaluation import Tracker

from pytracking.tracker.adnet.utils import *
from pytracking.tracker.adnet.train_ne import *
from pytracking.tracker.adnet.train_sl import *
from pytracking.tracker.adnet.train import *
from pytracking.tracker.adnet.synthetic import *