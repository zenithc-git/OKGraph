from openvocab_eval.prompts import _clean_name


def test_clean_name_articles():
    assert _clean_name("a dog") == "dog"
    assert _clean_name("an eagle") == "eagle"
    assert _clean_name("the cat") == "cat"
    assert _clean_name("Dog") == "Dog"
