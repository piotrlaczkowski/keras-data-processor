from pathlib import Path


class TestTimeSeriesDiagrams:
    """Tests to verify time series diagram generation."""

    def test_time_series_diagrams_exist(self):
        """Test that all time series diagrams have been generated."""
        # Define the expected diagrams
        expected_diagrams = [
            "basic_time_series.png",
            "time_series_with_lags.png",
            "time_series_moving_average.png",
            "time_series_differencing.png",
            "time_series_all_features.png",
        ]

        # Check both potential locations for the diagrams
        base_dirs = [
            Path("docs/features/imgs/models"),
            Path("generated_diagrams"),
        ]

        # Track which diagrams we've found
        found_diagrams = set()

        for base_dir in base_dirs:
            if not base_dir.exists():
                continue

            for diagram in expected_diagrams:
                diagram_path = base_dir / diagram
                if diagram_path.exists():
                    # Check that the file is not empty
                    assert (
                        diagram_path.stat().st_size > 0
                    ), f"Diagram {diagram} exists but is empty"
                    found_diagrams.add(diagram)

        # Assert that all expected diagrams were found
        missing = set(expected_diagrams) - found_diagrams
        assert not missing, f"Missing diagrams: {missing}"

    def test_time_series_diagrams_referenced_in_docs(self):
        """Test that the time series diagrams are referenced in the documentation."""
        docs_path = Path("docs/features/time_series_features.md")

        assert docs_path.exists(), "Time series features documentation file not found"

        # Read the documentation file content
        doc_content = docs_path.read_text()

        # Check for references to each diagram
        expected_references = [
            "imgs/models/basic_time_series.png",
            "imgs/models/time_series_with_lags.png",
            "imgs/models/time_series_moving_average.png",
            "imgs/models/time_series_differencing.png",
            "imgs/models/time_series_all_features.png",
        ]

        for reference in expected_references:
            assert (
                reference in doc_content
            ), f"Reference to {reference} not found in documentation"
