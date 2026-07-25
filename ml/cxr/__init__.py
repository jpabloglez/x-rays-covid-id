"""Chest radiograph dataset assembly, leakage gates, and modelling.

This package must never import from the web application, and the application
must never import from it. They communicate through exactly two artifacts: a
serialised model and a preprocessing config. Anything else couples research
code to request handling and guarantees a train/serve skew bug eventually.
"""

__version__ = "0.1.0"
