from PIL import Image

from jevgym import plot


def test_chart_has_no_footer_and_records_source_in_metadata(tmp_path):
    fig = plot.plt.figure()
    fig.text(0.5, 0.5, "body")
    run = tmp_path / "20990101T000000Z-x-dev"
    png, svg = plot.save(fig, tmp_path / "chart", [run], "jevgym plot test")
    source = f"{run} · jevgym plot test"
    assert Image.open(png).text["Source"] == source
    text = svg.read_text()
    assert f"<dc:source>{source}</dc:source>" in text
    assert "Source:" not in text  # nothing drawn on the image
