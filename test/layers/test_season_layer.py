import tensorflow as tf
from kdp.layers.season_layer import SeasonLayer


class TestSeasonLayer:
    """Test suite for SeasonLayer."""

    def test_season_encoding(self):
        """Test seasonal encoding of months."""
        layer = SeasonLayer()

        # Test different months
        dates = tf.constant(
            [
                [2023, 1, 1, 6],  # Winter
                [2023, 4, 1, 6],  # Spring
                [2023, 7, 1, 6],  # Summer
                [2023, 10, 1, 6],  # Fall
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check winter (December-February)
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])

        # Check spring (March-May)
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])

        # Check summer (June-August)
        assert tf.reduce_all(result[2, -4:] == [0, 0, 1, 0])

        # Check fall (September-November)
        assert tf.reduce_all(result[3, -4:] == [0, 0, 0, 1])

    def test_season_transition(self):
        """Test season transitions at boundary months."""
        layer = SeasonLayer()

        # Test boundary months
        dates = tf.constant(
            [
                [2023, 2, 28, 6],  # End of winter
                [2023, 3, 1, 6],  # Start of spring
                [2023, 5, 31, 6],  # End of spring
                [2023, 6, 1, 6],  # Start of summer
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check winter to spring transition
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])  # Still winter
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])  # Now spring

    def test_season_edge_months(self):
        """Test season assignment for edge case months."""
        layer = SeasonLayer()

        dates = tf.constant(
            [
                [2023, 12, 1, 0],  # December (Winter)
                [2023, 3, 1, 0],  # March (Spring)
                [2023, 6, 1, 0],  # June (Summer)
                [2023, 9, 1, 0],  # September (Fall)
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # Check correct season assignments
        assert tf.reduce_all(result[0, -4:] == [1, 0, 0, 0])  # Winter
        assert tf.reduce_all(result[1, -4:] == [0, 1, 0, 0])  # Spring
        assert tf.reduce_all(result[2, -4:] == [0, 0, 1, 0])  # Summer
        assert tf.reduce_all(result[3, -4:] == [0, 0, 0, 1])  # Fall

    def test_full_year_cycle(self):
        """Test season transitions through a full year."""
        layer = SeasonLayer()

        # Test middle month of each season
        dates = tf.constant(
            [
                [2023, 1, 15, 0],  # Mid-Winter
                [2023, 4, 15, 0],  # Mid-Spring
                [2023, 7, 15, 0],  # Mid-Summer
                [2023, 10, 15, 0],  # Mid-Fall
                [2024, 1, 15, 0],  # Back to Winter
            ],
            dtype=tf.int32,
        )

        result = layer(dates)

        # First winter and next winter should have same encoding
        assert tf.reduce_all(result[0, -4:] == result[4, -4:])
