## generate pb
```bash
# use betterproto, don't use pydantic before it's compatible with v2
# move to multi stage docker build to avoid submit it to git
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=grpc_servers/asr/pb/ pb/asr.proto
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=pb/ pb/asr.proto
```

## local test
```bash
cd asr
pytest -vv tests/server_test.py
```

## configuration
| environment  | default | comment                              |
| ------------ | ------- | ------------------------------------ |
| ASR_PROVIDER | whisper | whisper                              |
| ASR_MODEL    | tiny    | faster whisper compatible model name |
| ASR_HOST     | 0.0.0.0 | listen host                          |
| ASR_PORT     | 18909   | listen port                          |
