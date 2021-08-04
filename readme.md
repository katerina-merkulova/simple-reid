**trying to run fx at first time:**  
~~~
Traceback (most recent call last):
  File "/usr/local/bin/fx", line 8, in <module>
    sys.exit(entry())
  File "/usr/local/lib/python3.8/dist-packages/openfl/interface/cli.py", line 187, in entry
    command_group = import_module(module, package)
  File "/usr/lib/python3.8/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1014, in _gcd_import
  File "<frozen importlib._bootstrap>", line 991, in _find_and_load
  File "<frozen importlib._bootstrap>", line 975, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 671, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 848, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/usr/local/lib/python3.8/dist-packages/openfl/interface/aggregator.py", line 12, in <module>
    from openfl.federated import Plan
  File "/usr/local/lib/python3.8/dist-packages/openfl/federated/__init__.py", line 7, in <module>
    from .plan import Plan  # NOQA
  File "/usr/local/lib/python3.8/dist-packages/openfl/federated/plan/__init__.py", line 6, in <module>
    from .plan import Plan
  File "/usr/local/lib/python3.8/dist-packages/openfl/federated/plan/plan.py", line 13, in <module>
    from openfl.transport import AggregatorGRPCServer
  File "/usr/local/lib/python3.8/dist-packages/openfl/transport/__init__.py", line 6, in <module>
    from .grpc import AggregatorGRPCServer
  File "/usr/local/lib/python3.8/dist-packages/openfl/transport/grpc/__init__.py", line 6, in <module>
    from .server import AggregatorGRPCServer
  File "/usr/local/lib/python3.8/dist-packages/openfl/transport/grpc/server.py", line 12, in <module>
    from openfl.protocols import utils
  File "/usr/local/lib/python3.8/dist-packages/openfl/protocols/__init__.py", line 4, in <module>
    from .federation_pb2 import ModelProto, MetadataProto
  File "/usr/local/lib/python3.8/dist-packages/openfl/protocols/federation_pb2.py", line 23, in <module>
    create_key=_descriptor._internal_create_key,
AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
~~~

*solution:* https://stackoverflow.com/questions/61922334/how-to-solve-attributeerror-module-google-protobuf-descriptor-has-no-attribu  
`pip install --upgrade protobuf`

**ERROR TRACEBACK IS NOT FULL - i can't understand where is error at my code**
`[17:46:46] ERROR    Collaborator failed: list index out of range` is not enough!