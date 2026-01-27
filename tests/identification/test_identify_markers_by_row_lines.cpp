#include <catch2/catch_all.hpp>

#include "board.hpp"
#include "calibration.hpp"
#include "identification/board_circle/identification_circle.hpp"

namespace
{

// Helper to create a MarkerRing at given position with global_id
base::MarkerRing make_marker(int global_id, float row, float col)
{
    base::MarkerRing marker;
    marker.global_id_ = global_id;
    marker.row_ = row;
    marker.col_ = col;
    marker.width_ring_ = 10;
    marker.height_ring_ = 10;
    return marker;
}

// Create a simple board for testing
BoardCircleGrid make_test_board(int rows = 4, int cols = 5)
{
    BoardCircleGrid::Params params;
    params.rows = rows;
    params.cols = cols;
    params.spacing = 100.f;
    params.is_asymetric = false;
    params.radius = 10.f;
    params.padding_mm = Eigen::Vector2i(0, 0);
    return BoardCircleGrid(params);
}

}  // namespace

TEST_CASE("identify_new_markers_by_row_lines - complete rows identify untracked markers",
          "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    // Create markers: row 0 has 3 identified markers, row 1 has 2 identified + 1 unidentified
    std::vector<base::MarkerRing> markers;

    // Row 0: markers at cols 0, 1, 2 (all identified) - global_ids: 0, 1, 2
    markers.push_back(make_marker(0, 0.f, 0.f));
    markers.push_back(make_marker(1, 0.f, 100.f));
    markers.push_back(make_marker(2, 0.f, 200.f));

    // Row 1: markers at cols 0, 1 identified, col 2 unidentified - global_ids: 5, 6, should-be-7
    markers.push_back(make_marker(5, 100.f, 0.f));
    markers.push_back(make_marker(6, 100.f, 100.f));
    markers.push_back(make_marker(-1, 100.f, 200.f));  // unidentified, should become id=7

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    // The unidentified marker should now have global_id = 7 (row=1, col=2)
    REQUIRE(markers[5].global_id_ == 7);
}

TEST_CASE("identify_new_markers_by_row_lines - handles missing rows", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;

    // Row 0: 2 identified markers (enough to define line)
    markers.push_back(make_marker(0, 0.f, 0.f));
    markers.push_back(make_marker(2, 0.f, 200.f));

    // Row 2: 2 identified markers (row 1 is completely missing from tracking)
    markers.push_back(make_marker(10, 200.f, 0.f));    // id=10, row=2, col=0
    markers.push_back(make_marker(12, 200.f, 200.f));  // id=12, row=2, col=2

    // Unidentified marker on row 0
    markers.push_back(make_marker(-1, 0.f, 100.f));  // Should become id=1

    // Unidentified marker on row 2
    markers.push_back(make_marker(-1, 200.f, 100.f));  // Should become id=11

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    REQUIRE(markers[4].global_id_ == 1);
    REQUIRE(markers[5].global_id_ == 11);
}

TEST_CASE("identify_new_markers_by_row_lines - incomplete rows become complete", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;

    // Row 0: 2 markers identified (complete)
    markers.push_back(make_marker(0, 0.f, 0.f));
    markers.push_back(make_marker(2, 0.f, 200.f));

    // Row 1: complete row for reference
    markers.push_back(make_marker(5, 100.f, 0.f));
    markers.push_back(make_marker(6, 100.f, 100.f));
    markers.push_back(make_marker(7, 100.f, 200.f));

    // Unidentified marker that should be assigned based on row 0 line
    markers.push_back(make_marker(-1, 0.f, 100.f));  // Should become id=1

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    REQUIRE(markers[5].global_id_ == 1);
}

TEST_CASE("identify_new_markers_by_row_lines - respects distance threshold", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;

    // Row 0: 2 identified markers
    markers.push_back(make_marker(0, 0.f, 0.f));
    markers.push_back(make_marker(2, 0.f, 200.f));

    // Marker too far from any row line (should NOT be identified)
    // kLineDistanceThreshold is 10.0f, so 50 pixels away should not match
    markers.push_back(make_marker(-1, 50.f, 100.f));

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    REQUIRE(markers[2].global_id_ == -1);
}

TEST_CASE("identify_new_markers_by_row_lines - empty input", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();
    std::vector<base::MarkerRing> markers;

    // Should not crash with empty input
    REQUIRE_NOTHROW(identification::circlegrid::identify_new_markers_by_row_lines(markers, board));
}

TEST_CASE("identify_new_markers_by_row_lines - all unidentified markers", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;
    markers.push_back(make_marker(-1, 0.f, 0.f));
    markers.push_back(make_marker(-1, 0.f, 100.f));
    markers.push_back(make_marker(-1, 100.f, 0.f));

    // With no identified markers, no rows can be formed
    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    // All markers should still be unidentified
    REQUIRE(markers[0].global_id_ == -1);
    REQUIRE(markers[1].global_id_ == -1);
    REQUIRE(markers[2].global_id_ == -1);
}

TEST_CASE("identify_new_markers_by_row_lines - single marker per row", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;

    // Only 1 marker per row - not enough to form a line
    markers.push_back(make_marker(0, 0.f, 0.f));       // row 0, only 1 marker
    markers.push_back(make_marker(5, 100.f, 0.f));     // row 1, only 1 marker
    markers.push_back(make_marker(-1, 0.f, 100.f));    // unidentified on row 0
    markers.push_back(make_marker(-1, 100.f, 100.f));  // unidentified on row 1

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    // With only 1 marker per row, can't form lines, so unidentified markers stay unidentified
    REQUIRE(markers[2].global_id_ == -1);
    REQUIRE(markers[3].global_id_ == -1);
}

TEST_CASE("try_fill_missing_rows - finds entirely missing rows", "[identification][circlegrid]")
{
    BoardCircleGrid board = make_test_board();

    std::vector<base::MarkerRing> markers;

    // Row 0: 2 markers - this row will be in initial row_infos
    markers.push_back(make_marker(0, 0.f, 0.f));
    markers.push_back(make_marker(1, 0.f, 100.f));

    // Row 2: 2 markers - this row was NOT in initial row_infos (row 1 is skipped)
    // but try_fill_missing_rows should find it
    markers.push_back(make_marker(10, 200.f, 0.f));
    markers.push_back(make_marker(11, 200.f, 100.f));

    // Unidentified marker on row 2
    markers.push_back(make_marker(-1, 200.f, 200.f));  // Should become id=12

    identification::circlegrid::identify_new_markers_by_row_lines(markers, board);

    // The row 2 marker should now be identified
    REQUIRE(markers[4].global_id_ == 12);
}
