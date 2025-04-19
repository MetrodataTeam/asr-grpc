import asyncio
from enum import StrEnum
from io import BytesIO
import logging
from time import time
from typing import List

from grpclib import GRPCError
from grpclib import Status
from grpclib.health.service import Health
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server
from grpclib.utils import graceful_exit
import numpy as np
from pydantic import Field
from pydantic import model_validator
from pydantic import PositiveInt
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from pb.asr import AsrServiceBase
from pb.asr import Audio
from pb.asr import AudioResponse
from pb.asr import Info
from pb.asr import Segment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
for handler in logger.handlers:
  handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  )


class Provider(StrEnum):
  whisper = 'whisper'


DEFAULT_MODEL = {
  Provider.whisper: 'tiny',
}


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_prefix='asr_')
  provider: Provider = Field(Provider.whisper, description='asr provider')
  model: str = Field('', description='huggingface hub compatible model id')
  host: str = Field('0.0.0.0', description='listen host')
  port: int = Field(18909, description='listen port')
  num_workers: PositiveInt = Field(
    1, description='number of workers for transcribing audio'
  )
  # TODO(Deo): use it if we implement internal load balancing
  # concurrent: PositiveInt = Field(1, description='size of the tread pool')

  @model_validator(mode='after')
  def set_default_model(self):
    if not self.model:
      self.model = DEFAULT_MODEL[self.provider]
    return self


class ASRService(AsrServiceBase):
  def __init__(self, model: str, num_workers: PositiveInt = 1):
    from faster_whisper import WhisperModel

    self.model = WhisperModel(model, num_workers=num_workers)

  async def transcribe(self, request: Audio) -> AudioResponse:
    if not request.data:
      raise GRPCError(Status.FAILED_PRECONDITION, 'empty audio')
    start = time()
    if request.numpy_data:
      data = np.frombuffer(request.data, dtype=np.float32).copy()
    else:
      data = BytesIO(request.data)
    language = request.language or None
    vad_filter = request.vad_filter or False
    vad_parameters = {
      'threshold': 0.5,
      'neg_threshold': None,
      'min_speech_duration_ms': 0,
      'max_speech_duration_s': float('inf'),
      'min_silence_duration_ms': 2000,
      'speech_pad_ms': 400,
    }
    if vad_filter and request.vad_options is not None:
      # https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py#L37C4-L42C29
      vad_parameters = {
        'threshold': request.vad_options.threshold or 0.5,
        'neg_threshold': request.vad_options.neg_threshold or None,
        'min_speech_duration_ms': request.vad_options.min_speech_duration_ms
        or 0,
        'max_speech_duration_s': request.vad_options.max_speech_duration_s
        or float('inf'),
        'min_silence_duration_ms': request.vad_options.min_silence_duration_ms
        or 2000,
        'speech_pad_ms': request.vad_options.speech_pad_ms or 400,
      }
    segments, info = self.model.transcribe(
      data,
      initial_prompt=request.initial_prompt or None,
      language=language,
      vad_filter=vad_filter,
      vad_parameters=vad_parameters,
    )
    info = Info(language=info.language, probability=info.language_probability)
    segments = [
      Segment(
        start=i.start, end=i.end, text=i.text, no_speech_prob=i.no_speech_prob
      )
      for i in segments
    ]
    res = AudioResponse(
      segments=segments, info=info, text=''.join(i.text for i in segments)
    )
    time_span = time() - start
    if segments:
      logger.info(
        'finish transcribe %s in %.3f ms, duration: %.2fs, ratio %.2f',
        request.info or 'data',
        time_span * 1000,
        segments[-1].end,
        segments[-1].end / time_span,
      )
    else:
      logger.info(
        'finish transcribe %s in %.3f ms, no speech detected',
        request.info or 'data',
        time_span * 1000,
      )
    return res


def get_whisper_services(*args, **kwargs) -> List:
  return ServerReflection.extend([ASRService(*args, **kwargs), Health()])


async def serve(settings: Settings):
  server = Server(
    get_whisper_services(settings.model, num_workers=settings.num_workers)
  )
  with graceful_exit([server]):
    await server.start(settings.host, settings.port)
    logger.info(
      'listen on %s:%d, using %s %s',
      settings.host,
      settings.port,
      settings.provider,
      settings.model,
    )
    await server.wait_closed()
    logger.info('Goodbye!')


if __name__ == '__main__':
  _settings = Settings()
  asyncio.run(serve(_settings))
