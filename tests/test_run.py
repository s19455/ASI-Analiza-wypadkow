"""
Test basic pipeline execution.
"""

from kedro.framework.project import configure_project


class TestRun:
    def test_project_name(self):
        configure_project("crash_kedro")
