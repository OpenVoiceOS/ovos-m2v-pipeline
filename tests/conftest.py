"""Pytest configuration for the m2v test suite.

``test_live_fixture.py`` skips itself at *module* level via
``pytest.skip(allow_module_level=True)`` when ``OVOSCOPE_LIVE`` is unset. That
raises ``Skipped`` (a ``BaseException``) during import, which the ovoscope
pytest plugin's ``pytest_pycollect_makemodule`` wrapper only guards with
``except Exception`` — so the Skipped escapes, aborts the whole collection
session, and pytest reports "found no collectors" for *every* other test file
(including the ovoscope e2e suites). See the ovoscope plugin bug.

Until that is fixed upstream, ignore the live fixture during ordinary
(non-OVOSCOPE_LIVE) collection so the rest of the suite — unit tests and the
ovoscope end-to-end tests — collects normally. The dedicated live-test workflow
sets ``OVOSCOPE_LIVE=1`` and still collects it.
"""
import os

collect_ignore = []
if os.environ.get("OVOSCOPE_LIVE") != "1":
    collect_ignore.append("test_live_fixture.py")


import pytest


@pytest.fixture(autouse=True)
def _fresh_shared_models():
    """The embedding cache is process-wide; a mocked model from one test
    must not answer the next test's load."""
    from ovos_m2v_pipeline import clear_shared_models
    clear_shared_models()
    yield
    clear_shared_models()
