"""
Open-ended visual questions, answered by a local vision model.

Keyword commands (read, identify, describe) answer fixed questions
instantly. This handles everything else - "what am I wearing?", "what
colour is my bracelet?", "is there a power outlet?" - by sending the
current frame and the question to a vision-language model running
locally through Ollama.

MODEL CHOICE (tested 2-3 October):
  qwen2.5vl:3b      failed outright with a token-repeat error
  llava 7B          wrong shirt colour, missed a held phone, and
                    answered about an image it was never given
  moondream 1.8B    invented text on a bottle that was not there
  qwen3-vl:8b       accurate, but ~6 GB: sharing the 8 GB GPU with
                    detection, OCR and Whisper pushed it onto the CPU,
                    and a single answer took 95 seconds
  qwen3-vl:4b       accurate, ~3 GB, 2-5 seconds alongside the rest of
                    the application - the one in use here

HONESTY: the model sometimes states details it is not certain of. Answers
are spoken as "It looks like..." rather than as fact, and the prompt asks
it to admit uncertainty.
"""

import threading

import cv2
import ollama

MODEL = "qwen3-vl:4b"

SYSTEM_PROMPT = (
    "You are helping a blind person understand what a camera sees. "
    "Describe only what is actually visible. If you cannot tell, say so "
    "plainly. Answer in one or two short sentences, with no preamble."
)

# Maximum length of an answer, in tokens. Two short sentences fit easily.
MAX_ANSWER_TOKENS = 200


class VisionAsker:
    def __init__(self, model=MODEL):
        self.model = model
        self.busy = threading.Event()

        # Fail here, at startup, rather than when the user first asks:
        # the application can then report that the feature is unavailable
        # and carry on without it.
        ollama.list()

    def ask_in_background(self, frame, question, on_answer):
        """Send a frame and question to the model without blocking the loop.

        on_answer is called with (answer, None) on success, or
        (None, problem) when something went wrong.
        """
        if self.busy.is_set():
            on_answer(None, "I am still thinking about the last question")
            return

        self.busy.set()

        # The main loop replaces the frame constantly, so encode a
        # snapshot of this moment before handing it to the thread
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            self.busy.clear()
            on_answer(None, "I could not capture the image")
            return

        image_bytes = buffer.tobytes()

        def work():
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\n{question}",
                        "images": [image_bytes],
                    }],
                    # qwen3-vl reasons before answering, and that
                    # reasoning counts towards the length limit. On
                    # unclear questions it reasoned until the limit ran
                    # out and returned an empty answer - six times in one
                    # session. Turning reasoning off fixes that, and makes
                    # answers faster.
                    think=False,
                    options={"num_predict": MAX_ANSWER_TOKENS},
                )
                answer = response["message"]["content"].strip()

                # Safety net, in case reasoning text still arrives
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()

                if not answer:
                    on_answer(None, "I could not work that out. "
                                    "Please try asking again.")
                    return

                on_answer(answer, None)

            except Exception as error:
                print(f"Vision model error: {error}")
                on_answer(None, "the vision model did not respond")
            finally:
                self.busy.clear()

        threading.Thread(target=work, daemon=True).start()