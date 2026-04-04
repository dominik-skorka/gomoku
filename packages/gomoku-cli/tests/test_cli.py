from click.testing import CliRunner
from gomoku_cli.main import cli


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Gomoku" in result.output

    def test_play_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["play", "--help"])
        assert result.exit_code == 0
        assert "--black" in result.output
        assert "--white" in result.output
        assert "--board-size" in result.output

    def test_random_vs_random(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["play", "--black", "random", "--white", "random", "--board-size", "9"])
        assert result.exit_code == 0
        assert "wins" in result.output or "Draw" in result.output
