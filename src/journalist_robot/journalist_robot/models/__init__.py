"""Wrappers around the foundation models.

Heavy dependencies (torch, transformers, ...) are imported inside the
constructors so that the rest of the package can be imported and tested on
machines without a GPU.
"""
