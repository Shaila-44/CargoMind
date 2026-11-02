"""Shared helpers for the CargoMind shipment delay-risk EDA.

Modules
-------
config    : paths, class order, colours
io        : read-only loaders for the cleaned tables (no mutation of source files)
lineage   : raw-vs-processed comparison used to flag values that were filled during cleaning
stats     : effect sizes, tests and confidence intervals
plotting  : consistent matplotlib/seaborn styling and figure saving
features  : leakage-aware, as-of-departure integration of all tables to one row per shipment
"""
