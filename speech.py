import speech_recognition as sr
import time
def main():
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 2.0  
    microphone = sr.Microphone()
    output_log = []
    print("Voice Input Terminal App")
    print("Speak something, and I'll transcribe it. Say 'exit' to quit.")
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                output_log.append(f"You said: {text}")
                if text.lower() == "exit":
                    output_log.append("Exiting the app. Goodbye!")
                    break
                else:
                    output_log.append("You can say something else or 'exit' to quit.")
        except sr.UnknownValueError:
            output_log.append("Sorry, I couldn't understand that. Please try again.")
            break
        except sr.RequestError as e:
            output_log.append(f"Could not request results from the speech recognition service; {e}")
        except KeyboardInterrupt:
            output_log.append("\nExiting the app. Goodbye!")
            break
    print("\n".join(output_log))
if __name__ == "__main__":
    main()