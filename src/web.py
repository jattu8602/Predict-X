"""Serve the React dashboard and its API from one web service."""

from src.api import create_app
from src.data_loader import ROOT

app = create_app(dashboard_dir=ROOT / "dashboard" / "dist")
