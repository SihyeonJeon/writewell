from route.jev import FALLBACK, decide


def ans(form, fc, score, dc, noul):
    return {"form": {"type": "choice", "choice": form, "confidence": fc},
            "difficulty": {"type": "score", "score": score, "confidence": dc},
            "stakes": {"type": "noul", "noul": noul}}


def test_small_talk_goes_to_gemma():
    r = decide(ans("chat", 0.95, 0.1, 0.9, 0.02))
    assert (r.model, r.form) == ("gemma", "chat")


def test_risky_chat_goes_to_opus_high():
    r = decide(ans("chat", 0.95, 0.3, 0.9, 0.85))
    assert (r.model, r.effort) == ("opus", "high")


def test_longform_is_at_least_medium():
    r = decide(ans("longform", 0.9, 1.0, 0.8, 0.1))
    assert (r.model, r.effort, r.form) == ("opus", "medium", "longform")


def test_unsure_never_goes_to_gemma_and_uses_longform_gate():
    r = decide(ans("chat", 0.4, 0.1, 0.9, 0.0))
    assert r.model == "opus" and r.form == "longform"


def test_refusal_and_bad_shape_fall_back():
    a = ans("chat", 0.9, 0.1, 0.9, 0.0)
    a["stakes"] = {"type": "refusal"}
    assert decide(a) is FALLBACK
    assert decide({}) is FALLBACK
