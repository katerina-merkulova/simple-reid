**trying to run fx (both 1.1 and 1.2) at first time on python3.8 Ubuntu20:**  
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

**install openfl==1.2 on python3.7 Ubuntu16,20**:
~~~
Traceback (most recent call last):
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 584, in _build_master
    ws.require(__requires__)
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 901, in require
    needed = self.resolve(parse_requirements(requirements))
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 792, in resolve
    raise VersionConflict(dist, req).with_context(dependent_req)
pkg_resources.ContextualVersionConflict: (importlib-metadata 4.6.3 (/home/merkulov/venv_openfl/lib/python3.7/site-packages), Requirement.parse('importlib-metadata<4; python_version < "3.8.0"'), {'ipykernel'})

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/merkulov/venv_openfl/bin/fx", line 6, in <module>
    from pkg_resources import load_entry_point
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 3261, in <module>
    @_call_aside
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 3245, in _call_aside
    f(*args, **kwargs)
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 3274, in _initialize_master_working_set
    working_set = WorkingSet._build_master()
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 586, in _build_master
    return cls._build_from_requirements(__requires__)
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 599, in _build_from_requirements
    dists = ws.resolve(reqs, Environment())
  File "/home/merkulov/venv_openfl/lib/python3.7/site-packages/pkg_resources/__init__.py", line 787, in resolve
    raise DistributionNotFound(req, requirers)
pkg_resources.DistributionNotFound: The 'importlib-metadata<4; python_version < "3.8.0"' distribution was not found and is required by ipykernel
~~~

*solution:* `pip install 'importlib-metadata<4'`
