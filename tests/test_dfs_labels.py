from io import StringIO

from script_utils.prepare_dfs_labels import select_labels


def test_select_labels_retains_every_domain_and_the_original_fields():
    first = "AF-A0A1-F1-model_v4_TED01\tfirst\t-\n"
    second = "AF-A0A1-F1-model_v4_TED02\tsecond\t1.10.8\n"
    other = "AF-OTHER-F1-model_v4_TED01\tother\t2.40.50\n"
    output = StringIO()
    count = select_labels(StringIO(first + other + second), output, {"AF-A0A1-F1-model_v4"})
    assert count == 2
    assert output.getvalue() == first + second


def test_select_labels_does_not_replace_an_indexed_version():
    source = StringIO("AF-A0A1-F1-model_v6_TED01\t1.10.8\n")
    output = StringIO()
    assert select_labels(source, output, {"AF-A0A1-F1-model_v4"}) == 0
    assert not output.getvalue()
