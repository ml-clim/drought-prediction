collect_ignore = []

try:
    import paramiko
except ImportError:
    collect_ignore.append("exporters/test_gleam.py")

try:
    import xesmf
except ImportError:
    # pretty much all the preprocessors rely on
    # xesmf for regridding. They'll all fail without it
    # so this is an easier solutuon then adding an xfail
    # to all the modules
    collect_ignore.append("preprocess")

try:
    import BeautifulSoup
except ImportError:
    collect_ignore.append("exporters/test_chirps.py")

try:
    import xclim
except ImportError:
    collect_ignore.append("analysis/test_event_detector.py")

try:
    import bottleneck
except ImportError:
    collect_ignore.append("analysis/indices")

try:
    import climate_indices
except ImportError:
    collect_ignore.append("analysis/indices/test_spi.py")
