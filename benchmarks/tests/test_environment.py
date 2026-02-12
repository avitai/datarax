"""Tests for environment capture utility.

TDD: Write tests first, then implement.
Design ref: Section 6.4.8 of the benchmark report.
"""

from benchmarks.core.environment import capture_environment


class TestCaptureEnvironment:
    """Test capture_environment() function."""

    def test_returns_dict(self):
        """Test capture_environment returns a dict."""
        env = capture_environment()
        assert isinstance(env, dict)

    def test_contains_timestamp(self):
        """Test result contains a timestamp string."""
        env = capture_environment()
        assert "timestamp" in env
        assert isinstance(env["timestamp"], str)

    def test_contains_python_version(self):
        """Test result contains Python version."""
        env = capture_environment()
        assert "python_version" in env
        assert "3." in env["python_version"]  # Python 3.x

    def test_contains_jax_version(self):
        """Test result contains JAX version."""
        env = capture_environment()
        assert "jax_version" in env
        assert isinstance(env["jax_version"], str)

    def test_contains_platform_info(self):
        """Test result contains platform details."""
        env = capture_environment()
        assert "platform" in env
        platform = env["platform"]
        assert "backend" in platform
        assert "device_count" in platform
        assert "local_device_count" in platform
        assert "devices" in platform
        assert isinstance(platform["devices"], list)

    def test_contains_os_info(self):
        """Test result contains OS information."""
        env = capture_environment()
        assert "os" in env
        os_info = env["os"]
        assert "system" in os_info
        assert "release" in os_info
        assert "machine" in os_info

    def test_contains_git_commit(self):
        """Test result contains git commit hash (may be 'unknown')."""
        env = capture_environment()
        assert "git_commit" in env
        assert isinstance(env["git_commit"], str)

    def test_device_count_positive(self):
        """Test device count is at least 1."""
        env = capture_environment()
        assert env["platform"]["device_count"] >= 1

    def test_backend_is_string(self):
        """Test backend is a non-empty string."""
        env = capture_environment()
        assert isinstance(env["platform"]["backend"], str)
        assert len(env["platform"]["backend"]) > 0
