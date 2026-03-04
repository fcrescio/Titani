# Titani

Consumers Python per il sistema Ermete/Iris, gestiti con `uv`.

## Consumer disponibili

- **Teia**: riceve snapshot (`frame_available`), scarica il file, lo invia a una API LLM e pubblica sul websocket una descrizione testuale (`snapshot_description`).
- **Ceo**: riceve audio WebRTC, usa **WebRTC VAD** per rilevare rapidamente la voce, applica **Smart Turn v3** di `mlx-audio` (`mlx-community/smart-turn-v3`) per capire quando chi parla ha finito, trascrive il turno con **Qwen3 ASR** e sintetizza messaggi `say_to_user` (producer `teia`) con **Qwen3-TTS VoiceDesign**, reinviando l'audio su WebRTC (e su `.wav` in debug).
- **Crio**: riceve audio WebRTC e fa loopback basilare della traccia (pipeline torch/torchaudio placeholder).

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

Solo **Ceo**:

- `CEO_SILENCE_MS_BEFORE_ENDPOINT` (default: `300`, millisecondi di silenzio prima di interrogare Smart Turn)
- `CEO_SMART_TURN_THRESHOLD` (default: `0.5`, soglia per `predict_endpoint`)
- `CEO_ASR_MODEL` (default: `mlx-community/Qwen3-ASR-0.6B-8bit`)
- `CEO_ASR_LANGUAGE` (default: `Italian`)
- `CEO_DEBUG_MODE` (default: `false`, abilita heartbeat audio e dump segmenti per debug)
- `CEO_DEBUG_OUT_DIR` (default: `./ceo_debug`, directory in cui salvare i segmenti `.wav` inviati ad ASR)
- `CEO_DEBUG_HEARTBEAT_MS` (default: `2000`, frequenza log heartbeat audio in millisecondi)
- `CEO_TTS_MODEL` (default: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`)
- `CEO_TTS_LANGUAGE` (default: `Italian`)
- `CEO_TTS_INSTRUCT` (default: voce femminile adulta calda/naturale in italiano colloquiale)
- `CEO_TTS_STREAMING_INTERVAL` (default: `0.32`)
