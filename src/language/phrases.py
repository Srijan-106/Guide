"""
Everything VisionAssist says, in English and Hindi.

Keeping every spoken phrase here, rather than scattered through the code,
means a language can be added by translating one file - and a phrase
missing from one language falls back to English rather than crashing.

Hindi mode is Hinglish by design: sentences are in Hindi, but object
names stay in English ("मुझे दिख रहा है: bottle, chair"), which is how
most Hindi speakers say them anyway, and avoids translating all 365
object classes.

The Hindi phrases use the feminine first person ("पढ़ रही हूँ", "देखती
हूँ") because the Hindi voice, hi-IN-SwaraNeural, is female. If the voice
is changed to a male one such as hi-IN-MadhurNeural, those verb forms
must change too.
"""

PHRASES = {
    "en": {
        # Modes and language
        "ambient_mode":          "Ambient mode",
        "reading_mode":          "Reading mode",
        "reading_mode_hint":     "Reading mode. Press space to read text.",
        "switch_to_reading":     "Switch to reading mode first",
        "language_set":          "English mode",

        # Reading
        "reading":               "Reading",
        "still_reading":         "Still reading, please wait",
        "reading_failed":        "Sorry, reading failed",
        "no_text":               "No readable text found",

        # Identifying
        "identifying":           "Identifying",
        "i_see_object":          "I see {label}.",
        "might_be_object":       "This might be {label}.",
        "it_says":               "It says: {text}",
        "no_text_on_object":     "No readable text.",
        "cannot_identify":       "I cannot identify anything",

        # Describing
        "i_see_list":            "I see: {objects}",
        "nothing_recognised":    "I do not see anything I recognise",

        # Repeat
        "nothing_to_repeat":     "Nothing to repeat",

        # Visual questions
        "let_me_look":           "Let me look",
        "ask_full_question":     "Please ask a full question",
        "it_looks_like":         "It looks like {answer}",
        "vision_unavailable":    "Visual questions are not available",
        "vision_busy":           "Sorry, I am still thinking about the last question",
        "vision_capture":        "Sorry, I could not capture the image",
        "vision_empty":          "Sorry, I could not work that out. Please try asking again",
        "vision_no_response":    "Sorry, the vision model did not respond",

        # Voice commands
        "voice_unavailable":     "Voice commands are not available",
        "did_not_hear":          "I did not hear anything",
        "unknown_command":       "I heard {text}. I do not know that command.",
        "stopping":              "Stopping",

        # Failures
        "camera_problem":        "Camera problem",
        "camera_disconnected":   "Camera disconnected. Stopping.",
        "camera_back":           "Camera working again",
        "back_online":           "Back online",
    },

    "hi": {
        "ambient_mode":          "आसपास देखने का मोड",
        "reading_mode":          "पढ़ने का मोड",
        "reading_mode_hint":     "पढ़ने का मोड। टेक्स्ट पढ़ने के लिए स्पेस दबाइए।",
        "switch_to_reading":     "पहले पढ़ने वाला मोड चालू कीजिए",
        "language_set":          "हिंदी मोड",

        "reading":               "पढ़ रही हूँ",
        "still_reading":         "अभी पढ़ रही हूँ, थोड़ा रुकिए",
        "reading_failed":        "माफ़ कीजिए, पढ़ नहीं पाई",
        "no_text":               "पढ़ने लायक कोई टेक्स्ट नहीं मिला",

        "identifying":           "पहचान रही हूँ",
        "i_see_object":          "यह {label} है।",
        "might_be_object":       "यह शायद {label} है।",
        "it_says":               "इस पर लिखा है: {text}",
        "no_text_on_object":     "इस पर पढ़ने लायक कुछ नहीं लिखा है।",
        "cannot_identify":       "मैं कुछ पहचान नहीं पाई",

        "i_see_list":            "मुझे दिख रहा है: {objects}",
        "nothing_recognised":    "मुझे पहचानने लायक कुछ नहीं दिख रहा",

        "nothing_to_repeat":     "दोहराने के लिए कुछ नहीं है",

        "let_me_look":           "एक मिनट, देखती हूँ",
        "ask_full_question":     "कृपया पूरा सवाल पूछिए",
        "it_looks_like":         "ऐसा लगता है: {answer}",
        "vision_unavailable":    "तस्वीर के बारे में सवाल अभी उपलब्ध नहीं हैं",
        "vision_busy":           "माफ़ कीजिए, अभी पिछले सवाल पर सोच रही हूँ",
        "vision_capture":        "माफ़ कीजिए, तस्वीर नहीं ले पाई",
        "vision_empty":          "माफ़ कीजिए, समझ नहीं पाई। कृपया फिर से पूछिए",
        "vision_no_response":    "माफ़ कीजिए, विज़न मॉडल ने जवाब नहीं दिया",

        "voice_unavailable":     "आवाज़ से कमांड अभी उपलब्ध नहीं हैं",
        "did_not_hear":          "मुझे कुछ सुनाई नहीं दिया",
        "unknown_command":       "मैंने सुना: {text}. यह कमांड मुझे नहीं पता।",
        "stopping":              "बंद कर रही हूँ",

        "camera_problem":        "कैमरे में दिक्कत है",
        "camera_disconnected":   "कैमरा बंद हो गया। ऐप बंद कर रही हूँ।",
        "camera_back":           "कैमरा फिर से चल रहा है",
        "back_online":           "इंटरनेट वापस आ गया",
    },
}


def phrase(key, language, **values):
    """Return a phrase in the given language, filled in with values.

    Falls back to English if the phrase has not been translated, so a
    missing translation produces English speech rather than a crash.
    """
    table = PHRASES.get(language, PHRASES["en"])
    text = table.get(key, PHRASES["en"][key])
    return text.format(**values)