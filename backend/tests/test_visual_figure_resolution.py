import importlib.util
from pathlib import Path

PDF = Path('/mnt/data/complex data.pdf')


def _load_retrieval():
    # Import through package so relative imports resolve.
    import backend.rag_pipeline.retrieval as retrieval
    return retrieval


def test_figure_reference_parser_variants():
    r = _load_retrieval()
    assert r._extract_figure_reference('Explain Figure 5') == 5
    assert r._extract_figure_reference('What is Fig. 4?') == 4
    assert r._extract_figure_reference('Tell me figure number 2') == 2
    assert r._extract_figure_reference('Explain diagram no. 3') == 3
    assert r._extract_figure_reference('Explain Figure') is None


def test_real_pdf_figure_pages_are_resolved():
    r = _load_retrieval()
    assert PDF.exists(), PDF
    assert r._figure_pages_from_pdf(str(PDF), 1) == [3]
    assert 4 in r._figure_pages_from_pdf(str(PDF), 2)
    assert r._figure_pages_from_pdf(str(PDF), 3) == [13]
    assert r._figure_pages_from_pdf(str(PDF), 4) == [14]
    assert r._figure_pages_from_pdf(str(PDF), 5) == [15]
    assert r._figure_pages_from_pdf(str(PDF), 99) == []


def test_generic_figure_caption_pages_include_all_figures():
    r = _load_retrieval()
    pages = r._all_figure_caption_pages_from_pdf(str(PDF))
    assert set([3, 4, 13, 14, 15]).issubset(set(pages))
