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
  (`model`, `depo`, `transfer`) and names are never rewritten into
  look-alike terms.
- **Built-in software vocabulary.** Some 500 common terms (backend, pull
  request, Kubernetes…) ship with the app. While the model decodes a clip it
  is nudged toward these spellings and toward your own list (contextual
  biasing), which is what turns a misheard `bekleyen` back into `backendci`;
  afterwards the same list drives the fuzzy correction. Your list always
  wins over the built-in one.
- **Spelled-out acronyms and Turkish spelling.** Say "LLM'e" and Whisper
  writes the letter names it heard: `el eleme`. The worker turns a run of
  words that spells an acronym from your list or the built-in vocabulary
  back into the acronym, suffix included (`el eleme` → `LLM'e`, `ey pi ay` →
  `API`); on 217 stored texts it changed only the two that needed it. A short
  table of spellings that are wrong in every Turkish sentence (`yada` → `ya
  da`, `herşey` → `her şey`) is applied after that.
- **Learns from your corrections.** After a dictation is pasted, the app
  watches that text field through the Accessibility API. Fix a word and
  press Enter (or just pause) and the corrected spelling is added to your
  vocabulary on its own, like Wispr Flow's auto-learned dictionary — so the
  prompt, the biasing, the corrector and smart correction all know it next
  time. If the word the app wrote was itself a real Turkish word, it is not
  turned into a blind replacement: you may really mean it next time. Edits
  that are not hearing errors — two sentences joined, a different word
  chosen — are not learned at all: a correction has to sound like what it
  replaces.
- **Local memory with semantic search.** Your context sentences, your term
  list, learned corrections and dictations are kept in a SQLite database
  next to the models, indexed with SQLite's full-text search and with
  sentence embeddings from EmbeddingGemma 300M — the same quantized GGUF
  OpenClaw's memory search uses, run with llama.cpp on the CPU (9 ms a
  sentence; fallback: multilingual-e5-small through PyTorch). Whisper's prompt only holds about
  223 tokens, so the lists can be as long as you like: for *each* dictation
  the memory picks the context sentence and the terms that fit — the ones
  you use most, most recently, and in contexts closest to what you were just
  dictating. Smart correction gets your past corrections whose sentences are
  most similar to the new transcript, scored like OpenClaw does (vector
  similarity 0.7, keyword rank 0.3). Nothing leaves the machine.
- **A second listening.** Every clip under 30 s is decoded twice, the
  second time with beam search; the words the two passes disagree on go to
  smart correction as hints ("a second decoding heard *stoneları* as
  *sütunları*"). Measured on 16 clips with known text: every alternative
  decoding tried heard that word right where the first pass had it wrong,
  and beam search disagreed with the first pass in only two places. Nothing
  is replaced on this evidence alone; it costs about two seconds.
- **Smart correction (optional).** A local language model (Gemma 4 12B,
  4-bit, run with Apple's MLX; downloaded once) reads the finished sentence
  and names the words the recognizer got wrong: a software term written as
  a Turkish word, a non-word, or a real word that does not fit its sentence
  (`çalışırsan` for `çalışırken`). It answers with word pairs, never with a
  rewritten sentence, and a pair is applied only if the word occurs verbatim,
  the replacement sounds like it (same first sound, at most a third of the
  letters changed) and is a dictionary word or a vocabulary term — or if you
  taught exactly that pair. Measured on the user's own hearing errors it
  removed more than a third of them and changed none of 51 correct
  dictations; with the user's past corrections as examples it also carries a
  taught fix over to new sentences. On by default; it adds about four
  seconds per dictation. **Derin düzelt** in the menu asks a bigger model
  (Gemma 4 26B-A4B, loaded for that one request and dropped again; its
  first use downloads ~15 GB) to look at the last transcript once more and
  copies the result to the clipboard.
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
- Xcode 15 or later (to build the app)
- About 17 GB of free disk space: ~1.5 GB for the Python environment,
  ~2.9 GB for Whisper, ~0.3–0.5 GB for the memory's embedding model and
  ~11 GB for the smart-correction model (Gemma 4 12B through MLX). No
  Python needs to be installed: the app uses a Homebrew or python.org
  Python 3.10–3.12 if one is there and otherwise downloads its own.
  The Command Line Tools are only needed to compile llama.cpp for the
  EmbeddingGemma embeddings; without them the memory uses
  multilingual-e5-small, downloaded on first use.
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

The Python worker's model-free helpers (weight verification, the long-form
fallback heuristic) have unit tests too; they run with the app's own
environment:

```sh
cd Python && ~/Library/Application\ Support/WhisperHold/runtime/.venv/bin/python3 -m unittest test_whisper_worker test_term_bias test_term_llm test_memory test_acronyms test_dictionary test_gpu_lock test_benchmark
```

The correction layers are measured, not argued about. `Python/benchmark.py`
turns the memory into a test set — every correction you taught (the
sentence as Whisper wrote it, the word you meant) plus the dictations you
left alone — and runs the worker's text pipeline over it, reporting word
errors fixed against correct sentences damaged. Run it after changing a
model, a prompt or the gate:

```sh
cd Python && ~/Library/Application\ Support/WhisperHold/runtime/.venv/bin/python3 benchmark.py
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
2. **Wait for setup.** Everything else is automatic. The app creates its own
   Python environment in `~/Library/Application Support/WhisperHold/runtime`
   (downloading a Python if the machine has none), downloads Whisper into
   `~/Library/Application Support/WhisperHold/HuggingFace`, creates the
   memory database, and — as soon as Whisper is ready and dictation works —
   downloads the memory's embedding model and the smart-correction model in
   the background (~11 GB; the panel shows what is being prepared, and the
   worker is kept alive until the download is done). Depending on your
   connection the whole thing can take a while. Later launches load
   everything from disk.

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
| Model | Model status and verification details, restart the model, open the model folder, installed components with repair buttons |
| Kayıtlar (Recordings) | The recording archive: play, re-transcribe, see the latest text |
| Sözlük (Vocabulary) | Context sentence, technical terms, built-in software vocabulary, smart correction, learned corrections, exact replacements |
| Gelişmiş (Advanced) | Pre-roll, how long the microphone stays warm, minimum duration, silence threshold, archive folder |

### Vocabulary

- **Context sentences** — natural sentences about what you usually talk
  about, one per line, as many as you like. Whisper uses the selected one as
  a hint for style and topic.
- **Technical terms** — comma-separated spellings you want, such as
  `GitHub, Xcode, PyTorch`. They are added to the prompt and used to correct
  near-misses afterwards: `bild` → `build`, `komit` → `commit`, and glued
  Turkish suffixes get their apostrophe (`komponentleri` →
  `component'leri`, `GitHubdan` → `GitHub'dan`, `commitledim` →
  `commit'ledim`, `backendci` → `backend'ci`). A term the model has never
  seen in the prompt tends to come out as the nearest real Turkish word:
  `backendci` was heard as `bekleyen`, and listing `backend` alone did not
  change that — only listing `backendci` itself did. So list the forms you
  actually say, suffixed ones included; the sample list ships with
  `backendci` and `frontendci` for that reason. Before rewriting a word the corrector asks macOS's spelling
  dictionary for the dictation language; a word that exists there is left
  alone. This is what stops `model` from becoming `modal` or `depo` from
  becoming `deploy` — both happened with earlier versions. The one deliberate
  exception is a term's own Turkish respelling (`komponent`, `statik`): if you
  listed `component`, you get `component`. Azerbaijani uses the Turkish
  dictionary; automatic language detection checks all available ones.
- **Built-in vocabulary** (`Python/software_terms.txt`) — on by default. It
  is used for contextual biasing while decoding and for fuzzy correction
  afterwards, but more cautiously than your own list: a built-in term never
  turns a real Turkish word into its English spelling (`buton` stays `buton`
  unless *you* list `button`).
- **Smart correction** — on by default. The app downloads
  `mlx-community/gemma-4-12B-it-qat-4bit` (~11 GB) into the app's model
  cache on first launch and runs it with MLX next to Whisper, about 4 s per
  dictation;
  without `mlx-lm` the worker falls back to Qwen3-4B-Instruct through
  PyTorch. After each dictation the model is shown the sentence after a
  system prompt, ten fixed examples (half of them "nothing to change") and
  up to eight of your own past corrections whose sentences resemble it, and
  asked for a JSON list of `{wrong, right}` pairs. The worker applies a pair
  only when it passes
  the gate in `Python/term_llm.py`: the word occurs verbatim and is not
  already a term; the replacement carries no sentence punctuation, sounds
  like the word (same first sound, no more than a third of the letters — and
  never more than three — changed, with `c`/`k`, `ç`/`c`, `w`/`v` folded
  together) and every word of it is in macOS's Turkish or English dictionary
  or in the vocabulary. A pair you taught through learning skips the sound
  and dictionary checks: it is applied whenever the model proposes it in a
  fitting sentence. One more rule: a single real Turkish word is never
  turned into a term on the model's say-so (`Komite` → `Commit`, which the
  model once proposed on a synthetic test), only when taught; real-word
  fixes (`çalışırsan` → `çalışırken`) and split terms (`front ent` →
  `Frontend`) stay open. The gate was tuned on real proposals: it stops
  `Giral` → `Git`, `e-postada` → `endpoint'te` and `komite` → `commit` while
  passing `be kentçi` → `backendci`, `çıktığının` → `çıktının` and `RVD` →
  `RWD`. The
  first dictation after the worker restarts may run before the model has
  loaded; it is then skipped for that dictation. Set `WHISPERHOLD_LLM` to
  one or more MLX model ids (comma-separated) to try others; with several,
  their answers are merged, the first model winning where they disagree on
  a word.
- **Learned corrections** — on by default. The app keeps watching the field
  it pasted into for up to two minutes, reading its text through the
  Accessibility permission it already has for pasting. When one or two words
  of the pasted text have been replaced and you press Enter, switch fields or
  stop editing for a few seconds, the pair is learned: the corrected spelling
  joins your vocabulary (prompt, biasing, correction and smart correction),
  and the sentence it was corrected in is shown to the smart-correction model
  as an example. If the word the app had written exists in the dictionary of
  the dictation language (`bekleyen`), the pair is *not* applied as a blind
  replacement — saying that word on purpose must still work; the layers that
  read context decide case by case. A misspelling that is not a word
  (`endpoynt`) does become an exact replacement. A pair is only learned when
  it is plausibly a hearing error: no sentence punctuation inside, not a
  two-letter conjunction, and the two forms start with the same sound or
  share most of their letters (`Giral` → `geri al` yes, `işlediysen` →
  `yaptıysan` no — that was an edit, and teaching it to the model as an
  example only did harm). Chromium and Electron apps
  (Chrome, the Claude desktop app, Slack, VS Code) only build their
  accessibility tree when asked, so the app switches it on for the frontmost
  app before reading. Fields it still cannot read (terminals, password
  fields) are skipped and noted in the app log; for those, correct the text
  in the Recordings tab ("Son üretilen metin" is editable) and press
  **Düzeltmeyi öğret** — the same learning applies. The list is in the
  Vocabulary tab; any entry can be forgotten there.
- **Memory** (`memory.sqlite` in the app's support folder) — on by default.
  Context sentences (one per line, no limit), the hand-written term list
  (no limit), every learned correction and every dictation are stored, with
  an embedding once the embedding model is loaded (a few seconds after
  Whisper; rows stored before that are embedded afterwards). The worker
  composes each prompt within the token budget: the context sentence most
  related to your last dictations, then terms ranked by usage, recency and
  the similarity of the sentences they were seen in — hand-written terms
  carry a bonus, so they come first at equal relevance — then further
  context sentences if room is left. For smart correction the eight past
  corrections most similar to the transcript are shown to the model as
  examples — a correction you made once ("Cloudy" → "Claude") is applied in
  the next similar sentence. If the app cannot reach the worker (it is
  unloaded after 90 s idle), memory writes wait in a queue and are sent when
  it starts again. "Dikteleri hafızada tut" turns storage off; "Hafızayı
  temizle" deletes it.
- **Embedding model.** `Python/requirements-optional.txt` installs
  llama.cpp (`llama-cpp-python`, compiled from source with Metal) so the
  memory can run `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF` (~330 MB).
  If that package cannot be built on a machine, setup continues and the
  memory uses `intfloat/multilingual-e5-small` through PyTorch instead; the
  Vocabulary tab shows which one is active.
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
  packages and the models are downloaded from Hugging Face (Whisper, the
  embedding model, and the smart-correction model if you turn it on).
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

Decoding calls the model's `generate` directly, using Transformers' sequential
long-form algorithm rather than chunked inference, with greedy decoding and a
temperature ladder (`0.0, 0.4, 0.8`) that only engages when a segment's
compression ratio signals a repetition loop. The reasoning behind each setting
is documented in `Python/whisper_worker.py`. The worker does not go through
Transformers' speech-recognition pipeline: since 4.5x that pipeline silently
applies its own `num_beams=5` and `max_new_tokens=256` defaults when the
checkpoint sets neither, which is exactly the beam search measured below as
slower and less faithful.

A few settings are deliberately *not* used, because they were measured and
lost. Beam search (5 beams, the Whisper CLI default) was compared with greedy
decoding on the archived recordings under added white and pink noise down to
0 dB, at reduced gain and on a 23 s concatenation: greedy made fewer word
errors (8.1 % vs 10.3 %) and ran 2.2× faster; beam's extra errors were
rewrites of what was said (`abi` → `ağabey`). fp32 produced the same text as
fp16 and was slower. The prompt, on the other hand, earns its place: on ten
synthesized sentences with software terms it took term spelling from 11/20 to
20/20 and halved the word error rate. The *shape* of the prompt was measured
too, since Whisper was trained on preceding transcript text rather than
lists: a prompt made of the user's own past sentences (the ones richest in
terms) scored no better than no prompt at all (15.9 % word errors on the
synthesized sentences and two archived clips), the context sentence plus the
term list scored 9.3 %, and the two combined 12.1 %. The list stays.

Contextual biasing adds a fixed bonus to the score of every token that starts
or continues a vocabulary term, so the model prefers `backend` over the
look-alike `bek…` at the moment it decides. The strength (2.0) is the smallest
that fixed a real recording; it also lowered the word error rate on
synthesized term sentences from 9.1 % to 6.5 % and changed one of twelve
genuine Turkish sentences (the rare `rende` became `render`). On recordings
longer than one 30 s window the same bias was destructive at every strength
tried, so it is applied to single-window audio only.

Two safety nets cover the prompt's one measured failure. If the model answers
a clip with nothing but a subtitle artifact, the app decodes the clip once
more without the prompt before giving up. And on recordings longer than one
30 s window — where Transformers prepends the prompt to every window — the
worker checks for a window that is long but nearly empty (a 46 s test
recording once came back with 19 s of speech replaced by "Bu videoyu
izlediğiniz için teşekkür ederim.") and re-decodes without the prompt, which
transcribed that recording word for word. Healthy audio never takes either
second pass.

Whisper itself was challenged once more in October 2026 against Qwen3-ASR
1.7B (Qwen's own speech model, Turkish among its 52 languages). On the same
synthesized sentences and the user's archived recordings it made 2.6× the
word errors (15.9 % against Whisper's 6.2 %), has no prompt or biasing, and
turned "Hugging Face'ten" into something unprintable; Whisper large-v3 stays.
The remaining errors are measured with a text benchmark built from the user's
own transcripts: every hearing error they corrected by hand (the raw sentence
and the intended one) plus 51 dictations that must stay unchanged, scored as
word errors fixed against word errors introduced. The acronym and spelling
passes above fixed 5 of the 24 errors in that set without touching a single
correct sentence.

The same benchmark chose the smart-correction model and its prompt. The
previous model, Qwen3-4B-Instruct, fixed none of the 13 hearing errors with
any prompt. Gemma 4 12B (QAT 4-bit, MLX, 4.4 s per call) asked for word
pairs fixed `çıktığının` → `çıktının`, `çalışırsan` → `çalışırken` and `be
kentçi` → `backendci` and, behind the gate, changed none of the 51 correct
dictations; asked instead to return the corrected sentence it fixed one error
more but also "improved" three correct sentences (`benle` → `benimle`), which
is why the app asks for pairs. Qwen3.5-9B fixed less and damaged a correct
sentence under the same gate; Qwen3.8-27B matched Gemma on the hearing
errors and also repaired two Whisper typos the user had let through
(`dökümana`, `Mükerrel`), but at 9.8 s per call and 16 GB against 4.4 s and
11 GB. Gemma 4 26B-A4B, a mixture of experts with 4B active parameters, ran
at 1.4 s a call and fixed a different three (`RVD` → `RWD`, `yaptınca` →
`yaptın ki`, `be kentçi` → `backendci`), again touching nothing correct, and
the two Gemma sizes combined took the set from 24 word errors to 12 with
zero damage — but holding both next to Whisper costs about 30 GB, and the
app ships with the 12B alone (the pair is one `WHISPERHOLD_LLM` setting
away). Qwen3.6-35B-A3B (1.1 s a call) fixed one error and added nothing to
that pair. A synthetic set in the user's register (26 sentences with
Whisper-style mistakes, 18 that must not change) showed what the memory
adds: with the user's own past corrections as examples the 12B took 14
sentences to fully correct instead of 11 (the 26B 11 instead of 8), both
generalising a taught pair (`yaptınca` → `yaptın ki`) to new sentences; the
12B with the Swift corrector left 18 of 59 word errors, and its one
over-correction (`Komite` → `Commit`) is now caught by the gate.
The remaining errors are real-word substitutions no layer can tell from
intended speech (`eklemek` for `eklemeni`).

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
  Learning/       watches the pasted field and learns your corrections
Tests/            unit tests for the text pipeline
Python/
  whisper_worker.py   model loading and transcription
  term_bias.py        contextual biasing toward the vocabulary
  term_llm.py         optional smart correction with a local language model
  memory.py           SQLite memory, full-text search, embeddings, prompt selection
  software_terms.txt  built-in software vocabulary
  install_runtime.sh  creates the virtual environment
  requirements.txt    pinned Python dependencies
  requirements-optional.txt  llama.cpp for EmbeddingGemma (setup continues without it)
Config/           code signing settings
project.yml       XcodeGen project definition
```

## Troubleshooting

- **The shortcut does nothing.** Check that Input Monitoring is granted in
  System Settings → Privacy & Security. With an ad-hoc signed build, every
  rebuild needs the permissions again; see [Code signing](#code-signing).
- **Text appears on the clipboard but is not pasted.** Accessibility
  permission is missing, or a password field (secure input) is active.
- **Setup stops at the Python step.** The app looks for a Python 3.10–3.12
  from Homebrew or python.org and otherwise downloads a standalone one
  (verified by checksum); if that download fails (offline, blocked), choose
  **Modeli yeniden başlat** (Restart model) on the Model tab once you are
  online, or install Python with `brew install python@3.11`.
- **"llama.cpp (EmbeddingGemma): kurulamadı — Command Line Tools yok"** in
  Settings → Model → Bileşenler (Components). llama.cpp compiles from source
  and needs Xcode's Command Line Tools; the installer skips it rather than
  trigger Apple's install dialog mid-setup, and the memory runs on the
  fallback embedding model meanwhile (dictation and smart correction are
  unaffected). Click **Command Line Tools'u kur** (Apple's installer opens),
  then **Eksik paketleri kur** (Install missing packages): the worker stops,
  the package is built, the worker restarts, and the memory moves to
  EmbeddingGemma, re-embedding everything the fallback model had indexed
  (the two models' vectors are not comparable). The same panel lists every
  component the installer found — Python, PyTorch, Transformers, MLX, PyObjC,
  llama.cpp — with versions, from `runtime/components.json`.
- **What did the app do?** Everything the worker reports, model loading,
  smart-correction and memory status, and learned corrections are written to
  `~/Library/Application Support/WhisperHold/whisperhold.log` (capped at
  1 MB); macOS's unified log does not reliably show this menu bar app.
- **"Python beklenmedik şekilde kesildi" right after pressing the shortcut.**
  Fixed in the current build: PyTorch's MPS backend is not thread-safe, and
  the smart-correction and embedding models used to be moved to the GPU on a
  background thread while Whisper was decoding the first dictation after a
  worker restart, which tripped a Metal assertion
  (`_status < MTLCommandBufferStatusCommitted`). All GPU work now goes
  through one lock (`Python/gpu_lock.py`); a model finishing its load waits
  for the dictation to finish instead. The memory's embedding model runs on
  the CPU for a related reason: llama.cpp's Metal backend once hung the
  worker at start-up, spinning in its GPU residency wait while another
  process was loading a model, with the lock held. Should a load ever hang
  again, smart correction gives up on it after two minutes rather than
  holding every dictation. `Python/test_gpu_lock.py` fails the
  moment a GPU call is added outside that lock, and should the worker ever
  die mid-dictation for another reason, the app restarts it and transcribes
  the same audio again instead of losing the dictation.
- **Starting over.** Quit the app and delete
  `~/Library/Application Support/WhisperHold`. The Python environment and
  the model are recreated on next launch.

## Uninstall

Delete the app, then remove its data:

```sh
rm -rf ~/Library/Application\ Support/WhisperHold
rm -rf ~/Documents/WhisperHold
```

The first folder also holds the memory database and the app log.

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
