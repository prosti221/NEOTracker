from pytracking.evaluation.environment import EnvSettings
from pathlib import Path

def local_env_settings():
    settings = EnvSettings()
    root = Path(__file__).parent.parent.parent.parent

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.youtubevos_dir = ''

    settings.synthetic_path = root / 'datasets/synthetic'    # Where the synthetic dataset is stored.
    settings.network_path = root / 'networks'    # Where tracking networks are stored.
    settings.result_plot_path = root / 'pytracking/pytracking/result_plots'    # Where to store result plots.
    settings.results_path = root / 'pytracking/pytracking/tracking_results'    # Where to store tracking results.
    settings.segmentation_path = root / 'pytracking/pytracking/segmentation_results'    # Where to store segmentation results.
    settings.vot_path = root / 'datasets/vot2016/sequences'    # Where the VOT sequences are stored.

    return settings

