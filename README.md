# WhisperHold

Hold-to-talk dictation for macOS that runs entirely on your Mac. Hold a
shortcut, speak, let go — the text is typed into whatever app has focus.

Speech recognition uses OpenAI's **official, unmodified Whisper large-v3**
checkpoint. There is no Core ML, MLX or ONNX conversion and no quantization:
the FP16 `model.safetensors` published on Hugging Face is loaded as-is with
PyTorch + Transformers and runs on the Apple Silicon GPU (MPS). After the
one-time download, no audio or text ever leaves your machine, and no API key
is needed.

> **Note:** the app's interface is currently in Turkish. Dictation itself
> works in Turkish, English, Azerbaijani, German and Russian, or with
> automatic language detection.

## Features

- **Push-to-talk from anywhere.** Default shortcut is holding `⌃ Control` +
  `⌘ Command`; a single modifier (such as right `⌥ Option`), another
  modifier combination or modifier + key can be recorded instead. Pressing
  another key while holding a modifier-only shortcut cancels the recording,
  so normal shortcuts like `⌃⌘F` keep working.
- **Full-quality model.** The 32 + 32 layer large-v3 checkpoint, verified by
  SHA-256 against Hugging Face's content address on first load.
- **Vocabulary that sticks.** A context sentence and a list of technical terms
  are fed to Whisper as a prompt, terms are fuzzy-corrected after
  transcription (including glued Turkish suffixes: `komponentleri` →
  `component'leri`, `commitledim` → `commit'ledim`), and exact replacements
  fix words the model keeps getting wrong. The fuzzy pass asks macOS's own
  spelling dictionary first, so real words of the dictation language
  (`model`, `depo`, `transfer`) are never rewritten into look-alike terms.
- **Hallucination and loop guards.** Known subtitle artifacts ("Thanks for
  watching", "Altyazı M.K.") are dropped instead of pasted — as a whole clip
  or as a stray sentence inside a longer dictation — runaway repetition is
  re-decoded at a higher temperature and collapsed, and a watchdog stops a
  stuck request from freezing the app.
- **Recording archive and re-transcription.** Optionally keep the last five
  dictations as WAV files, listen to them, and re-run any of them after
  changing the language or vocabulary.
- **Light on resources.** The model is unloaded after 90 seconds idle. The next
  recording starts instantly while the model reloads from disk in parallel, so
  the first words aren't cut off.
- **Menu bar app.** Recording indicator with a live level meter, sound cues,
  input device selection and launch at login.

## Requirements

- macOS 14 Sonoma or later
- An Apple Silicon Mac (M1 or newer). The pinned PyTorch release ships no
  Intel macOS builds.
- Xcode 15 or later
- Python 3 — 3.11 is preferred: `brew install python@3.11`
- About 3.5 GB of free disk space: ~0.6 GB for the Python environment and
  ~2.9 GB for the model
- [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`),
  only if you change `project.yml`

## Build and run

```sh
git clone https://github.com/Anar-Mamedov/WhisperHold.git
cd WhisperHold
open WhisperHold.xcodeproj
```

Then press **Run** in Xcode. Or build from the command line:

```sh
xcodebuild -project WhisperHold.xcodeproj -scheme WhisperHold \
  -configuration Release -derivedDataPath build
open build/Build/Products/Release/WhisperHold.app
```

`project.yml` is the source of truth for the Xcode project. After editing it,
run `xcodegen generate`.

### Tests

The text pipeline (term correction, spelling guard, hallucination and
repetition filters) has unit tests. They compile the pure Swift sources
directly, so running them never launches the app or asks for permissions:

```sh
xcodebuild test -project WhisperHold.xcodeproj -scheme WhisperHold \
  -destination 'platform=macOS'
```

### Code signing

Out of the box the app is signed ad-hoc ("Sign to Run Locally"), so no Apple
Developer account is needed. The catch: macOS ties the Microphone, Input
Monitoring and Accessibility grants to the app's signature, and an ad-hoc
signature changes with every build — so each rebuild asks for those
permissions again.

To keep permissions across builds, sign with your own certificate (a free
Apple ID works):

```sh
cp Config/Local.xcconfig.example Config/Local.xcconfig
```

and set `DEVELOPMENT_TEAM` to your team ID. `Config/Local.xcconfig` is
git-ignored.

## First launch

1. **Grant three permissions.** The menu bar panel walks you through them:
   - **Microphone** — to record your voice.
   - **Input Monitoring** — to notice the shortcut being held and released.
   - **Accessibility** — to send `⌘V` to the app you are typing in.

   macOS may require restarting the app after granting Accessibility.
2. **Wait for setup.** The app creates its own Python virtual environment in
   `~/Library/Application Support/WhisperHold/runtime`, then downloads the
   model from Hugging Face into
   `~/Library/Application Support/WhisperHold/HuggingFace`. Depending on your
   connection this can take a while; progress is shown in the menu bar panel.
   Later launches load everything from disk.

When the model is ready, **Settings → Model** shows the source URL, the actual
path and size of the safetensors file, the layer count, and the PyTorch and
Transformers versions in use.

## Usage

Hold the shortcut while you speak and release it when you are done. The audio
is transcribed locally and pasted at the cursor. The text is also left on the
clipboard, so nothing is lost if pasting fails — for example in password
fields, where macOS blocks synthetic key presses.

Recordings that are too short or too quiet are discarded without being sent to
the model (both thresholds are adjustable).

### Settings

Labels in the app are in Turkish; the English names are given alongside.

| Tab | What it controls |
| --- | --- |
| Genel (General) | Shortcut, dictation language, input device, sound cues, recording indicator, launch at login |
| Model | Model status and verification details, restart the model, open the model folder |
| Kayıtlar (Recordings) | The recording archive: play, re-transcribe, see the latest text |
| Sözlük (Vocabulary) | Context sentence, technical terms, exact replacements |
| Gelişmiş (Advanced) | Pre-roll, how long the microphone stays warm, minimum duration, silence threshold, archive folder |

### Vocabulary

- **Context sentence** — a natural sentence about what you usually talk about.
  Whisper uses it as a hint for style and topic.
- **Technical terms** — comma-separated spellings you want, such as
  `GitHub, Xcode, PyTorch`. They are added to the prompt and used to correct
  near-misses afterwards: `bild` → `build`, `komit` → `commit`, and glued
  Turkish suffixes get their apostrophe (`komponentleri` →
  `component'leri`, `GitHubdan` → `GitHub'dan`, `commitledim` →
  `commit'ledim`). Before rewriting a word the corrector asks macOS's spelling
  dictionary for the dictation language; a word that exists there is left
  alone. This is what stops `model` from becoming `modal` or `depo` from
  becoming `deploy` — both happened with earlier versions. The one deliberate
  exception is a term's own Turkish respelling (`komponent`, `statik`): if you
  listed `component`, you get `component`. Azerbaijani uses the Turkish
  dictionary; automatic language detection checks all available ones.
- **Replacements** — exact `wrong → right` pairs for words the model
  consistently gets wrong.

### Recording archive and re-transcription

If the folder `~/Documents/WhisperHold` exists, every dictation is saved
there as a WAV file and only the five most recent are kept. Create it with the
**Arşivi aç** (Open archive) button on the Recordings tab, or with `mkdir`;
delete the folder to turn the archive off.

The Recordings tab lists each recording with its date, duration, how many times
it has been transcribed, and the latest text the model produced for it. If a
transcript came out wrong, play the recording and choose **Yeniden çevir**
(Re-transcribe). The audio is sent through again with the *current* language,
context sentence and vocabulary, so you can tweak the context and retry as
often as you like. The audio file is never changed — only the text stored next
to it is updated.

A re-transcribed result is copied to the clipboard and shown on screen rather
than pasted, because at that moment the app's own window has focus. Switch to
your target app and press `⌘V`.

The menu bar panel also has **Yeniden çevir**, which re-transcribes the most
recent recording, and **Kayıtlar…**, which opens the Recordings tab.

Two attempts on the same recording do not always produce identical text. If
the model falls into a repetition loop on some segment, that segment is decoded
again at a higher temperature, which is random — and that is exactly why
re-transcribing can help. Healthy audio produces the same text every time.

## Privacy

- Audio is recorded and transcribed on your Mac. Nothing is sent to OpenAI or
  any other server, and no account or API key is involved.
- Network access is used only during setup: `pip` installs the pinned Python
  packages and the model is downloaded from Hugging Face.
- Recordings are written to disk only if you enable the archive folder, and
  only the five most recent are kept.

## How it works

```
 Hotkey (CGEventTap) ──► AudioRecorder (16 kHz mono, pre-roll)
                                │
                                ▼
                     trim trailing silence, gate short/quiet clips
                                │
                                ▼
 PythonWhisperEngine ──JSON over stdin/stdout──► whisper_worker.py
                                                  (PyTorch + Transformers, MPS)
                                │
                                ▼
 TextPostProcessor: hallucination filter, repetition collapse,
                    term correction, replacements
                                │
                                ▼
 TextInjector: clipboard + synthetic ⌘V into the focused app
```

The native app is SwiftUI/AppKit. Transcription runs in a long-lived Python
worker process that loads the model once and talks to the app over
line-delimited JSON. The app installs and manages that worker itself; you never
need to run Python by hand.

Decoding uses Transformers' sequential long-form algorithm rather than chunked
inference, with a temperature ladder (`0.0, 0.4, 0.8`) that only engages when a
segment's compression ratio signals a repetition loop. The reasoning behind
each setting is documented in `Python/whisper_worker.py`.

A few settings are deliberately *not* used, because they were measured and
lost. Beam search (5 beams, the Whisper CLI default) was compared with greedy
decoding on the archived recordings under added white and pink noise down to
0 dB, at reduced gain and on a 23 s concatenation: greedy made fewer word
errors (8.1 % vs 10.3 %) and ran 2.2× faster; beam's extra errors were
rewrites of what was said (`abi` → `ağabey`). fp32 produced the same text as
fp16 and was slower. The prompt, on the other hand, earns its place: on ten
synthesized sentences with software terms it took term spelling from 11/20 to
20/20 and halved the word error rate.

Two safety nets cover the prompt's one measured failure. If the model answers
a clip with nothing but a subtitle artifact, the app decodes the clip once
more without the prompt before giving up. And on recordings longer than one
30 s window — where Transformers prepends the prompt to every window — the
worker checks for a window that is long but nearly empty (a 46 s test
recording once came back with 19 s of speech replaced by "Bu videoyu
izlediğiniz için teşekkür ederim.") and re-decodes without the prompt, which
transcribed that recording word for word. Healthy audio never takes either
second pass.

### Project layout

```
Sources/
  App/            app entry point, controller, menu bar panel
  Audio/          recording, device selection, silence trimming, archive
  Hotkey/         global shortcut monitor and shortcut model
  Indicator/      floating recording indicator
  Output/         clipboard + paste injection
  Permissions/    microphone / input monitoring / accessibility checks
  Settings/       preferences and settings window
  Transcription/  Python worker client, text post-processing, spelling guard
Tests/            unit tests for the text pipeline
Python/
  whisper_worker.py   model loading and transcription
  install_runtime.sh  creates the virtual environment
  requirements.txt    pinned Python dependencies
Config/           code signing settings
project.yml       XcodeGen project definition
```

## Troubleshooting

- **The shortcut does nothing.** Check that Input Monitoring is granted in
  System Settings → Privacy & Security. With an ad-hoc signed build, every
  rebuild needs the permissions again; see [Code signing](#code-signing).
- **Text appears on the clipboard but is not pasted.** Accessibility
  permission is missing, or a password field (secure input) is active.
- **"Python 3 bulunamadı" (Python 3 not found) during setup.** Install Python
  with `brew install python@3.11` and choose **Modeli yeniden başlat** (Restart
  model) on the Model tab.
- **Starting over.** Quit the app and delete
  `~/Library/Application Support/WhisperHold`. The Python environment and
  the model are recreated on next launch.

## Uninstall

Delete the app, then remove its data:

```sh
rm -rf ~/Library/Application\ Support/WhisperHold
rm -rf ~/Documents/WhisperHold
```

## Contributing

Issues and pull requests are welcome. Some areas where help would be
especially useful:

- English localization of the interface
- Packaging signed, notarized releases
- More hallucination phrases for other languages

The decoding settings in `Python/whisper_worker.py` were chosen by measuring
real recordings, and the comments there explain the trade-offs. If you change
them, please describe how you measured the effect in your pull request. Changes
to the text pipeline should come with a case in `Tests/`.

## License

This project is released under the [MIT License](LICENSE).

The Whisper large-v3 model weights are published by OpenAI under the Apache
2.0 license and are downloaded from
[Hugging Face](https://huggingface.co/openai/whisper-large-v3) at first launch;
they are not part of this repository. Python dependencies (PyTorch,
Transformers and others) are installed from PyPI under their own licenses.
