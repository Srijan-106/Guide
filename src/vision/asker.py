"""
Open-ended visual questions, answered by a local vision model.

Keyword commands (read, identify, describe) answer fixed questions
instantly. This handles everything else - "what am I wearing?", "what
colour is my bracelet?", "where are the objects?" - by sending the
current frame and the question to a vision-language model running
locally through Ollama.

MODEL CHOICE (tested 2-5 October):
  qwen2.5vl:3b      failed outright with a token-repeat error
  llava 7B          wrong shirt colour, missed a held phone, and
                    answered about an image it was never given
  moondream 1.8B    invented text on a bottle that was not there
  qwen3-vl:8b       accurate, but ~6 GB: sharing the 8 GB GPU with
                    detection, OCR and Whisper pushed it onto the CPU,
                    and a single answer took 95 seconds
  qwen3-vl:4b       accurate and fast, but this tag is the THINKING
                    variant: it always reasons before answering, the
                    reasoning used up the length limit and returned
                    empty answers, and it kept the GPU busy long enough
                    to make the video stutter
  qwen3-vl:4b-instruct
                    the same model without the reasoning step - the one
                    in use here

HINDI: asked to look at the image and answer in Hindi at the same time,
the 4B model became inaccurate - it invented objects ("a cooler near the
window", "traffic number boards") that were not there. Its English
answers to the same questions were correct. So Hindi answers are made in
two steps: the model answers in English, which it does accurately, and
that answer is then translated. English is the "pivot" language.

The translation is done by a separate, dedicated translation model
running locally - see src/language/translator.py. When the 4B vision
model translated its own answers it changed facts ("a red bracelet on the
right wrist" became "on the knee of the right hand"), and Google's free
translation endpoint refused every request. If translation is not
possible, the English answer is spoken instead: an accurate answer in
English is better than a wrong one in Hindi.

An earlier version put an example Hindi sentence in the prompt to show
the Hinglish style. The model copied it into every answer, whatever the
question - so no examples are given.

HONESTY: the model sometimes states details it is not certain of. Answers
are spoken as "It looks like..." rather than as fact, and the prompt asks
it to admit uncertainty.
"""

import threading

import cv2
import ollama

MODEL = "qwen3-vl:4b-instruct"

# Always answers in English - see the note on Hindi above.
#
# "Address the user as you": the camera faces the person asking, and
# without this the model described them as "he" or "the person". Its
# translation then became "उसने" or "लोग ने" instead of "आपने".
#
# "Answer only what was asked": asked the colour of the wall, it also
# described the user's clothes.
VISION_PROMPT = (
    "You are helping a blind person understand what a camera sees. The "
    "camera is pointed at the person asking, so address them as 'you'. "
    "Describe only what is actually visible. If you cannot tell, say so "
    "plainly. Answer only what was asked, in one or two short sentences, "
    "with no preamble. The question may be in Hindi; always answer in "
    "English."
)

# A question that has not been answered within this many seconds is
# abandoned. Without a limit, one slow answer kept the model "busy", and
# every question after it was refused - five in a row in one test.
TIMEOUT_SECONDS = 30

# Maximum length of an answer, in tokens. Two short sentences fit easily.
MAX_ANSWER_TOKENS = 200


class VisionAsker:
    def __init__(self, model=MODEL, translator=None):
        """translator - turns English answers into Hindi; see
        src/language/translator.py. Without one, Hindi mode speaks the
        English answer."""
        self.model = model
        self.translator = translator
        self.busy = threading.Event()
        self.client = ollama.Client(timeout=TIMEOUT_SECONDS)

        # Fail here, at startup, rather than when the user first asks:
        # the application can then report that the feature is unavailable
        # and carry on without it. Checking only that Ollama was running
        # was not enough - with the model missing, the application
        # started normally and failed on every question.
        installed = self._installed_models()
        if model not in installed:
            raise RuntimeError(
                f"the vision model '{model}' is not installed. "
                f"Install it with:  ollama pull {model}"
            )

    def _installed_models(self):
        """Names of the models Ollama has downloaded.

        Newer versions of the ollama package return objects, older ones
        dictionaries, so both are handled.
        """
        response = self.client.list()
        models = getattr(response, "models", None)
        if models is None:
            models = response.get("models", [])

        names = set()
        for entry in models:
            if isinstance(entry, dict):
                name = entry.get("model") or entry.get("name")
            else:
                name = getattr(entry, "model", None)
            if name:
                names.add(name)
        return names

    def _translate_to_hindi(self, english):
        """Translate an answer into Hindi, or return None if that fails."""
        if self.translator is None:
            print("  [vision] no translator - speaking English")
            return None
        try:
            return self.translator.translate(english)
        except Exception as error:
            print(f"  [vision] translation failed, speaking English: {error}")
            return None

    def _chat(self, content, images=None, max_tokens=MAX_ANSWER_TOKENS):
        """One request to the model; returns its reply as plain text."""
        message = {"role": "user", "content": content}
        if images:
            message["images"] = images

        response = self.client.chat(
            model=self.model,
            messages=[message],
            options={"num_predict": max_tokens},
        )
        answer = response["message"]["content"].strip()

        # Safety net, in case reasoning text ever arrives
        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        return answer

    def ask_in_background(self, frame, question, on_answer, language="en"):
        """Send a frame and question to the model without blocking the loop.

        language - "en" or "hi": the language the answer should be in.

        on_answer is called with (answer, None) on success, or
        (None, problem) when something went wrong. The problem is a short
        code - "busy", "capture", "empty" or "no_response" - rather than
        an English sentence, so the caller can say it in either language.
        """
        if self.busy.is_set():
            on_answer(None, "busy")
            return

        self.busy.set()

        # The main loop replaces the frame constantly, so encode a
        # snapshot of this moment before handing it to the thread
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            self.busy.clear()
            on_answer(None, "capture")
            return

        image_bytes = buffer.tobytes()

        def work():
            try:
                # Step 1: look at the image and answer, always in English
                answer = self._chat(f"{VISION_PROMPT}\n\n{question}",
                                    images=[image_bytes])
                if not answer:
                    on_answer(None, "empty")
                    return

                # Step 2, Hindi mode only: translate the English answer.
                # If translation is unavailable, the English answer is
                # spoken - accurate English beats inaccurate Hindi.
                if language == "hi":
                    print(f"  [vision] English answer: {answer}")
                    hindi = self._translate_to_hindi(answer)
                    if hindi:
                        answer = hindi

                on_answer(answer, None)

            except Exception as error:
                print(f"Vision model error: {error}")
                on_answer(None, "no_response")
            finally:
                self.busy.clear()

        threading.Thread(target=work, daemon=True).start()