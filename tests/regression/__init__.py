"""WxFactory regression suite.

A small, self-contained package that runs every runnable config in ``config/``
sequentially, produces outputs + plots, compares against a previous run
(SSIM + data L2, report-only) and publishes a browsable gallery.

Entry point: :mod:`tests.regression.run_regression`
(run as ``python tests/regression/run_regression.py`` or
``python -m tests.regression.run_regression``).
"""
