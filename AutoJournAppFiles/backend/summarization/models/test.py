from summarization.models.bart import BartSummarizationModel

def test_remove_links():
    model = BartSummarizationModel()
    assert model.preprocess("http://wokeosdaskd.com Andrew: I am going there.") == \
        " Andrew: I am going there."

def test_remove_tags():
    model = BartSummarizationModel()
    assert model.preprocess("Andrew: @mariam I am going there.") == \
        "Andrew:  I am going there."

def test_process_emojis():
    model = BartSummarizationModel()
    assert model.preprocess("""Andrew: I am going there \ud83e\udd23 .""") == \
        "Andrew: I am going there 🤣 ."