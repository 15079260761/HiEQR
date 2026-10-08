from decgc.data.temporal_parser import parse_time, temporal_compatible


def test_temporal_parser_preserves_unknown():
    assert parse_time("2007-02-27")["granularity"] == "day"
    assert parse_time("27 February 2007")["normalized"] == "2007-02-27"
    assert parse_time("2007-02")["granularity"] == "month"
    assert parse_time("n/a")["normalized"] is None
    assert parse_time("December 2015")["normalized"] == "2015-12"
    assert parse_time("February 27, 2007")["normalized"] == "2007-02-27"
    assert parse_time("Jun 25th, 2006")["normalized"] == "2006-06-25"
    assert parse_time("25th Jun 2006")["normalized"] == "2006-06-25"
    assert parse_time("2012-06-8")["normalized"] == "2012-06-08"
    assert parse_time("2012-6")["normalized"] == "2012-06"
    assert parse_time("Jun, 2011")["normalized"] == "2011-06"
    assert parse_time("13.10.2010")["normalized"] == "2010-10-13"


def test_temporal_compatibility_uses_complete_month_boundaries():
    assert temporal_compatible("2015-12-15", ["December 2015"], "equal")
    assert not temporal_compatible("2015-11-30", ["December 2015"], "equal")
    assert temporal_compatible("2015-11-30", ["December 2015"], "before")
    assert temporal_compatible("2016-01-01", ["December 2015"], "after")
