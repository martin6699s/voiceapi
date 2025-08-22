from typing import *
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Query, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import logging
from pydantic import BaseModel, Field
import uvicorn
from voiceapi.tts import TTSResult, start_tts_stream, TTSStream
from voiceapi.asr import start_asr_stream, ASRStream, ASRResult, load_asr_engine
import numpy as np
class ASRRequest(BaseModel):
    samplerate: int = Field(16000, title="Sample Rate", description="The sample rate of the audio.")
    model: str = Field(None, title="ASR Model", description="ASR model name.")
    lang: str = Field(None, title="ASR Language", description="ASR language.")


import logging
import argparse
import os

app = FastAPI()
logger = logging.getLogger(__file__)


# HTTP API for ASR
@app.post("/httpasr", description="Recognize speech from uploaded audio file.")
async def asr_recognize(
    file: UploadFile = File(..., description="Audio file (wav/pcm)"),
    samplerate: int = Query(16000, title="Sample Rate", description="The sample rate of the audio."),
    model: str = Query(None, title="ASR Model", description="ASR model name."),
    lang: str = Query(None, title="ASR Language", description="ASR language.")
):
    """
    HTTP API for ASR: upload audio file and get recognition result.
    """
    # 可选参数覆盖 args
    asr_args = args
    if model:
        asr_args.asr_model = model
    if lang:
        asr_args.asr_lang = lang

    try:
        asr_stream: ASRStream = await start_asr_stream(samplerate, asr_args)
        if not asr_stream:
            return {"success": False, "error": "failed to start ASR stream"}

        audio_bytes = await file.read()
        logger.info(f"[httpasr] uploaded audio bytes: {len(audio_bytes)}")
        if len(audio_bytes) < 3200:
            return {"success": False, "error": "audio too short or empty"}
        
        logger.info(f"[httpasr] WAV文件并解码为PCM")
        
        # 检查是否为WAV文件并获取原始16bit PCM bytes
        import io
        import wave
        pcm_bytes = None
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                if wf.getnchannels() != 1:
                    return {"success": False, "error": "wav must be mono (1 channel)"}
                if wf.getsampwidth() != 2:
                    return {"success": False, "error": "wav must be 16bit"}
                if wf.getframerate() != samplerate:
                    return {"success": False, "error": f"wav samplerate must be {samplerate}"}
                pcm_bytes = wf.readframes(wf.getnframes())
        except Exception as e:
            logger.warning(f"Not a valid wav file, treat upload as raw PCM bytes: {e}")
            pcm_bytes = audio_bytes

        if not pcm_bytes or len(pcm_bytes) < 4:
            return {"success": False, "error": "audio data empty or too short"}

        # 尝试使用一次性识别（绕过 VAD/流处理），对整个音频做离线解码
        try:
            recognizer = load_asr_engine(samplerate, asr_args)
            stream_tmp = recognizer.create_stream()
            # 将原始 int16 bytes 转为 float32 samples
            pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
            samples = pcm_int16.astype(np.float32) / 32768.0
            stream_tmp.accept_waveform(samplerate, samples)
            # 对于在线/离线识别器都尝试 decode
            try:
                # online recognizer may require decode while is_ready
                if hasattr(recognizer, 'is_ready') and recognizer.is_ready(stream_tmp):
                    while recognizer.is_ready(stream_tmp):
                        recognizer.decode_stream(stream_tmp)
                else:
                    recognizer.decode_stream(stream_tmp)
            except Exception:
                # some recognizers may not implement decode_stream in same way
                try:
                    recognizer.decode_stream(stream_tmp)
                except Exception:
                    pass

            # 获取识别结果
            text = None
            if hasattr(recognizer, 'get_result'):
                r = recognizer.get_result(stream_tmp)
                # get_result may return string or object
                if isinstance(r, str):
                    text = r
                elif hasattr(r, 'text'):
                    text = r.text
            else:
                # stream_tmp.result may exist
                try:
                    text = stream_tmp.result.text.strip()
                except Exception:
                    text = None

            if text:
                return {"success": True, "text": text}
        except Exception as e:
            logger.warning(f"one-shot recognition failed: {e}")

        # 作为后备，仍发送给流式引擎（原来的方式）
        await asr_stream.write(pcm_bytes)
        try:
            result: ASRResult = await asyncio.wait_for(asr_stream.read(), timeout=15)
        except asyncio.TimeoutError:
            await asr_stream.close()
            return {"success": False, "error": "recognition timeout, please check audio content and format"}
        await asr_stream.close()
        if not result or not getattr(result, "text", None):
            return {"success": False, "error": "no recognition result"}
        return {"success": True, "text": result.text}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket,
                        samplerate: int = Query(16000, title="Sample Rate",
                                                description="The sample rate of the audio."),):
    await websocket.accept()

    asr_stream: ASRStream = await start_asr_stream(samplerate, args)
    if not asr_stream:
        logger.error("failed to start ASR stream")
        await websocket.close()
        return

    async def task_recv_pcm():
        while True:
            pcm_bytes = await websocket.receive_bytes()
            if not pcm_bytes:
                return
            await asr_stream.write(pcm_bytes)

    async def task_send_result():
        while True:
            result: ASRResult = await asr_stream.read()
            if not result:
                return
            await websocket.send_json(result.to_dict())
    try:
        await asyncio.gather(task_recv_pcm(), task_send_result())
    except WebSocketDisconnect:
        logger.info("asr: disconnected")
    finally:
        await asr_stream.close()


@app.websocket("/tts")
async def websocket_tts(websocket: WebSocket,
                        samplerate: int = Query(16000,
                                                title="Sample Rate",
                                                description="The sample rate of the generated audio."),
                        interrupt: bool = Query(True,
                                                title="Interrupt",
                                                description="Interrupt the current TTS stream when a new text is received."),
                        sid: int = Query(0,
                                         title="Speaker ID",
                                         description="The ID of the speaker to use for TTS."),
                        chunk_size: int = Query(1024,
                                                title="Chunk Size",
                                                description="The size of the chunk to send to the client."),
                        speed: float = Query(1.0,
                                             title="Speed",
                                             description="The speed of the generated audio."),
                        split: bool = Query(True,
                                            title="Split",
                                            description="Split the text into sentences.")):

    await websocket.accept()
    tts_stream: TTSStream = None

    async def task_recv_text():
        nonlocal tts_stream
        while True:
            text = await websocket.receive_text()
            if not text:
                return

            if interrupt or not tts_stream:
                if tts_stream:
                    await tts_stream.close()
                    logger.info("tts: stream interrupt")

                tts_stream = await start_tts_stream(sid, samplerate, speed, args)
                if not tts_stream:
                    logger.error("tts: failed to allocate tts stream")
                    await websocket.close()
                    return
            logger.info(f"tts: received: {text} (split={split})")
            await tts_stream.write(text, split)

    async def task_send_pcm():
        nonlocal tts_stream
        while not tts_stream:
            # wait for tts stream to be created
            await asyncio.sleep(0.1)

        while True:
            result: TTSResult = await tts_stream.read()
            if not result:
                return

            if result.finished:
                await websocket.send_json(result.to_dict())
            else:
                for i in range(0, len(result.pcm_bytes), chunk_size):
                    await websocket.send_bytes(result.pcm_bytes[i:i+chunk_size])

    try:
        await asyncio.gather(task_recv_text(), task_send_pcm())
    except WebSocketDisconnect:
        logger.info("tts: disconnected")
    finally:
        if tts_stream:
            await tts_stream.close()


class TTSRequest(BaseModel):
    text: str = Field(..., title="Text",
                      description="The text to be converted to speech.",
                      examples=["Hello, world!"])
    sid: int = Field(0, title="Speaker ID",
                     description="The ID of the speaker to use for TTS.")
    samplerate: int = Field(16000, title="Sample Rate",
                            description="The sample rate of the generated audio.")
    speed: float = Field(1.0, title="Speed",
                         description="The speed of the generated audio.")


@ app.post("/tts",
           description="Generate speech audio from text.",
           response_class=StreamingResponse, responses={200: {"content": {"audio/wav": {}}}})
async def tts_generate(req: TTSRequest):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    tts_stream = await start_tts_stream(req.sid, req.samplerate, req.speed,  args)
    if not tts_stream:
        raise HTTPException(
            status_code=500, detail="failed to start TTS stream")

    r = await tts_stream.generate(req.text)
    return StreamingResponse(r, media_type="audio/wav")


if __name__ == "__main__":
    models_root = './models'

    for d in ['.', '..', '../..']:
        if os.path.isdir(f'{d}/models'):
            models_root = f'{d}/models'
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--addr", type=str,
                        default="0.0.0.0", help="serve address")

    parser.add_argument("--asr-provider", type=str,
                        default="cpu", help="asr provider, cpu or cuda")
    parser.add_argument("--tts-provider", type=str,
                        default="cpu", help="tts provider, cpu or cuda")

    parser.add_argument("--threads", type=int, default=2,
                        help="number of threads")

    parser.add_argument("--models-root", type=str, default=models_root,
                        help="model root directory")

    parser.add_argument("--asr-model", type=str, default='sensevoice',
                        help="ASR model name: zipformer-bilingual, sensevoice, paraformer-trilingual, paraformer-en, fireredasr")

    parser.add_argument("--asr-lang", type=str, default='zh',
                        help="ASR language, zh, en, ja, ko, yue")

    parser.add_argument("--tts-model", type=str, default='vits-zh-hf-theresa',
                        help="TTS model name: vits-zh-hf-theresa, vits-melo-tts-zh_en, kokoro-multi-lang-v1_0")

    args = parser.parse_args()

    if args.tts_model == 'vits-melo-tts-zh_en' and args.tts_provider == 'cuda':
        logger.warning(
            "vits-melo-tts-zh_en does not support CUDA fallback to CPU")
        args.tts_provider = 'cpu'

    app.mount("/", app=StaticFiles(directory="./assets", html=True), name="assets")

    logging.basicConfig(format='%(levelname)s: %(asctime)s %(name)s:%(lineno)s %(message)s',
                        level=logging.INFO)
    uvicorn.run(app, host=args.addr, port=args.port)
