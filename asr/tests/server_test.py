from functools import partial
import wave

from grpclib import GRPCError
from grpclib import Status
from grpclib.health.v1.health_grpc import HealthStub
from grpclib.health.v1.health_pb2 import HealthCheckRequest
from grpclib.health.v1.health_pb2 import HealthCheckResponse
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest
from grpclib.testing import ChannelFor
import numpy as np
import pytest
from server import get_whisper_services
from server import Settings

from pb.asr import AsrServiceStub
from pb.asr import Audio

approx = partial(pytest.approx, abs=1e-5)


@pytest.fixture(scope='module', autouse=True)
def anyio_backend():
  return 'asyncio'


async def test_asr_service():
  settings = Settings()
  services = get_whisper_services(settings.model)
  with open('tests/test.ogg', 'rb') as f:
    data = f.read()
  async with ChannelFor(services) as channel:
    stub = AsrServiceStub(channel)
    health = HealthStub(channel)
    reflection = ServerReflectionStub(channel)

    # empty data
    with pytest.raises(GRPCError) as e:
      response = await stub.transcribe(Audio(data=b'', info='test.ogg'))
    assert e.value.status == Status.FAILED_PRECONDITION

    response = await stub.transcribe(Audio(info='test.ogg', data=data))
    assert response.info.language == 'zh'
    assert response.info.probability > 0.9
    # NOTICE: result is not stable, just check the format
    assert len(response.segments) == 1
    assert response.text

    # health
    response = await health.Check(HealthCheckRequest())
    assert response.status == HealthCheckResponse.SERVING

    # reflection
    response = await reflection.ServerReflectionInfo(
      [ServerReflectionRequest(file_containing_symbol='ASR')]
    )
    assert len(response) == 1
    # TODO(Deo): it's not found at the moment
    #   https://github.com/danielgtaylor/python-betterproto/issues/443
    # assert response[0].name == ''
    # assert response[0].package == ''

    # TODO(Deo): make it work
    # unary_unary as stream control
    # async with stub.transcribe.open() as stream:
    #   for _ in range(2):
    #     await stream.send_message(Audio(info='test.ogg', data=data))
    #     response = await stream.recv_message()
    #     assert response.info.language == 'zh'
    #     assert response.info.probability > 0.9
    #     assert response.segments == [
    #         Segment(start=0.0,
    #                 end=4.0,
    #                 text='银车是',
    #                 no_speech_prob=approx(0.14626722037792206))
    #     ]
    #     assert response.text == '银车是'


async def test_asr_from_numpy():
  settings = Settings()
  services = get_whisper_services(settings.model)
  with open('tests/test.wav', 'rb') as f:
    file_data = f.read()
  with wave.open('tests/test.wav', 'rb') as wf:
    audio_data = wf.readframes(wf.getnframes())
    if wf.getsampwidth() == 2:
      audio_array = np.frombuffer(audio_data, dtype=np.int16)
    elif wf.getsampwidth() == 4:
      audio_array = np.frombuffer(audio_data, dtype=np.int32)
    else:
      raise ValueError(f'Unsupported sample width: {wf.getsampwidth()} bytes')
    # https://stackoverflow.com/questions/76448210/how-to-feed-a-numpy-array-as-audio-for-whisper-model
    data = (audio_array.astype(np.float32) / 32768.0).tobytes()
  async with ChannelFor(services) as channel:
    stub = AsrServiceStub(channel)

    # file data
    response = await stub.transcribe(Audio(info='test.wav', data=file_data))
    assert response.info.language == 'zh'
    assert response.info.probability > 0.9
    assert len(response.segments) == 1
    segment = response.segments[0]
    assert segment.start == 0.0
    assert segment.end == 4.0
    # 锄禾日当午
    assert len(segment.text) == 5, segment.text
    assert response.text == segment.text

    # numpy data
    response = await stub.transcribe(
      Audio(info='test.wav', data=data, numpy_data=True, vad_filter=True)
    )
    assert response.info.language == 'zh'
    assert response.info.probability > 0.9
    assert len(response.segments) == 1
    segment = response.segments[0]
    assert segment.start == approx(2.1600000858306885)
    assert segment.end == approx(4.159999847412109)
    assert len(segment.text) == 5, segment.text
    assert response.text == segment.text


async def test_asr_language():
  settings = Settings()
  services = get_whisper_services(settings.model, num_workers=2)
  with open('tests/test.ogg', 'rb') as f:
    data = f.read()
  async with ChannelFor(services) as channel:
    stub = AsrServiceStub(channel)

    response = await stub.transcribe(
      Audio(info='test.ogg', data=data, language='yue')
    )
    assert response.info.language == 'yue'
    assert response.info.probability == 1.0
    assert response.segments == []
    assert response.text == ''
