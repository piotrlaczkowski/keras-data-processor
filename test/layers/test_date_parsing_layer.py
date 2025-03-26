import pytest
import tensorflow as tf
from kdp.layers.date_parsing_layer import DateParsingLayer


class TestDateParsingLayer:
    """Test suite for DateParsingLayer."""

    def test_date_parsing_valid_formats(self):
        """Test parsing of valid date formats."""
        layer = DateParsingLayer()

        # Test different valid formats
        dates = tf.constant(
            [
                ["2025-01-17"],
                ["2024/06/16"],
                ["2023-12-31"],
            ]
        )

        result = layer(dates)
        assert result.shape == (3, 4)  # [batch_size, (year, month, day, day_of_week)]

        # Check first date (2023-01-15)
        assert result[0][0] == 2025  # year
        assert result[0][1] == 1  # month
        assert result[0][2] == 17  # day
        assert result[0][3] == 5  # day of week (Friday)

    def test_date_parsing_invalid_formats(self):
        """Test handling of invalid date formats."""
        layer = DateParsingLayer()

        # Test invalid formats
        invalid_dates = tf.constant(
            [
                ["20230115"],  # No separators
                ["2023-99-15"],  # Invalid month
                ["2023-01-32"],  # Invalid day
            ]
        )

        with pytest.raises(tf.errors.InvalidArgumentError):
            layer(invalid_dates)

    def test_date_parsing_edge_cases(self):
        """Test edge cases for date parsing."""
        layer = DateParsingLayer()

        edge_dates = tf.constant(
            [
                ["2023-01-01"],  # Start of year
                ["2023-12-31"],  # End of year
                ["2024-02-29"],  # Leap year
            ]
        )

        result = layer(edge_dates)
        assert result.shape == (3, 4)

        # Check New Year's Day
        assert result[0][0] == 2023  # year
        assert result[0][1] == 1  # month
        assert result[0][2] == 1  # day
        assert result[0][3] == 0  # day of week (Sunday)
