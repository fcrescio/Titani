# Titani

Consumers Python per il sistema Ermete/Iris, gestiti con `uv`.

## Consumer disponibili

- **Teia**: riceve snapshot (`frame_available`) via WebSocket dal backend, scarica il file, lo invia a una API LLM e pubblica sul data channel WebRTC `cmd` una descrizione testuale (`snapshot_description`). Inoltre riceve `speaker_turn_completed`, filtra trascrizioni troppo corte/spurie, inoltra i turni validi all'LLM mantenendo uno storico conversazionale e reinvia la risposta come `say_to_user`.
- **Ceo**: riceve audio WebRTC, usa **WebRTC VAD** per rilevare rapidamente la voce, applica **Smart Turn v3** di `mlx-audio` (`mlx-community/smart-turn-v3`) per capire quando chi parla ha finito, trascrive il turno con **Qwen3 ASR** e sintetizza messaggi `say_to_user` (producer `teia`) con **Qwen3-TTS Base (voice cloning via reference audio)**, reinviando l'audio su WebRTC usando il sample rate della traccia client (con resampling automatico, e dump `.wav` in debug). I messaggi applicativi (`say_to_user`, `speaker_turn_completed`, `ping/pong`) passano sul data channel `cmd`.
- **Crio**: riceve audio WebRTC e fa loopback basilare della traccia (pipeline torch/torchaudio placeholder), con segnali applicativi gestiti sul data channel `cmd`.

## Setup rapido

```bash
uv sync
```

Esecuzione:

```bash
uv run teia
uv run ceo
uv run crio
```

## Variabili ambiente principali

- `ERMETE_WS` (default: `wss://alveare.metallize.it:8080/v1/ws?role=consumer`)
- `ERMETE_HTTP_BASE` (default: `https://alveare.metallize.it:8080`)
- `ERMETE_PSK_HEADER` (default: `X-Ermete-PSK`)
- `ERMETE_PSK`
- `FRAMES_OUT_DIR` (default: `./frames_downloaded`)

Solo **Teia**:

- `LLM_API_KEY` (oppure `OPENAI_API_KEY`)
- `LLM_MODEL` (default: `gpt-4.1-mini`)
- `LLM_BASE_URL` (default: `https://api.openai.com/v1`)
- `TEIA_LLM_MIN_TURN_CHARS` (default: `8`, lunghezza minima per considerare valida una trascrizione `speaker_turn_completed`)
- `TEIA_LLM_HISTORY_MAX_TURNS` (default: `20`, numero massimo di turni conversazionali mantenuti in memoria)

Solo **Ceo**:

- `CEO_PRE_ROLL_MS` (default: `250`, audio pre-roll in millisecondi incluso all'inizio turno)
- `CEO_INGRESS_PROFILE` (default: `balanced`, profilo VAD ingress predefinito: `balanced`, `noisy`, `fast`)
- `CEO_ADVANCED_TUNING` (default: `false`, abilita override granulari VAD via env dedicati)
- `CEO_START_SPEECH_CHUNKS` (override *solo* con `CEO_ADVANCED_TUNING=1`, frame consecutivi speech richiesti per avviare un turno)
- `CEO_SPEECH_MAJORITY_RATIO` (override *solo* con `CEO_ADVANCED_TUNING=1`, quota minima di subchunk speech per considerare un frame parlato)
- `CEO_SPEECH_SUBCHUNK_MIN_COUNT` (override *solo* con `CEO_ADVANCED_TUNING=1`, minimo assoluto di subchunk speech per frame)
- `CEO_VAD_MIN_RMS` (override *solo* con `CEO_ADVANCED_TUNING=1`, soglia RMS minima opzionale per ridurre falsi positivi VAD)
- `CEO_SILENCE_MS_BEFORE_ENDPOINT` (default: `300`, millisecondi di silenzio prima di entrare in endpoint candidate)
- `CEO_MAX_SILENCE_MS_FORCE_COMMIT` (default: `1500`, timeout hard di silenzio continuo per chiudere turno)
- `CEO_TRAILING_SILENCE_PAD_MS` (default: `200`, pad massimo di silenzio finale incluso nel segmento)
- `CEO_DEBUG_DUMP_WAV_ENABLED` (default: `false`, abilita dump WAV debug dei turni smart-turn committati)
- `CEO_DEBUG_DUMP_WAV_DIR` (default: `./ceo_debug/smart_turn`, directory dei dump WAV smart-turn)
- `CEO_ASR_MODEL` (default: `mlx-community/Qwen3-ASR-0.6B-8bit`)
- `CEO_ASR_LANGUAGE` (default: `Italian`)
- `CEO_DEBUG_MODE` (default: `false`, abilita heartbeat audio e dump segmenti per debug)
- `CEO_DEBUG_OUT_DIR` (default: `./ceo_debug`, directory in cui salvare i segmenti `.wav` inviati ad ASR)
- `CEO_DEBUG_HEARTBEAT_MS` (default: `2000`, frequenza log heartbeat audio in millisecondi)
- `CEO_DEBUG_VAD_TRACE` (default: `false`, abilita trace dettagliato decisioni VAD/stato turn in log)
- `CEO_DEBUG_VAD_TRACE_EVERY_CHUNKS` (default: `25`, ogni quanti chunk forzare un log anche in assenza di speech)
- `CEO_DEBUG_VAD_TRACE_JSONL` (default: `false`, salva anche trace strutturato in `ceo_debug/vad_trace.jsonl`)
- `CEO_TTS_MODEL` (default: `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16`)
- Nota: per il backend TTS corrente (`mlx_audio.tts`) la lingua non è configurabile via `generate()`: eventuali variabili legacy di lingua vengono ignorate con warning esplicito nei log.
- `CEO_TTS_REF_AUDIO` (default: vuoto, path al file `.wav` di riferimento per il voice cloning)
- `CEO_TTS_REF_TEXT` (default: vuoto, trascrizione del file audio di riferimento)
- `CEO_TTS_STREAMING_INTERVAL` (default: `0.32`)

Profili ingress predefiniti (`CEO_INGRESS_PROFILE`):

| Profilo | `start_speech_chunks` | `speech_majority_ratio` | `speech_subchunk_min_count` | `vad_min_rms` | Trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
| `balanced` | 10 | 0.50 | 2 | 0.00 | Compromesso generale tra latenza e robustezza. |
| `noisy` | 12 | 0.65 | 3 | 0.02 | Più robusto al rumore/falsi positivi, ma con avvio turno più lento. |
| `fast` | 6 | 0.40 | 1 | 0.00 | Minima latenza di aggancio turno, ma più sensibile a rumore e spike. |

Nota: con `CEO_ADVANCED_TUNING=0` (default) gli override granulari vengono ignorati con warning nei log e prevale sempre il profilo selezionato.

### Nota migrazione CEO

Dalle versioni precedenti, le variabili `CEO_ENDPOINT_RETRY_MS` e `CEO_SMART_TURN_THRESHOLD` non sono più utilizzate: ora sono deprecate/rimosse e vengono ignorate se ancora presenti nell'ambiente.
