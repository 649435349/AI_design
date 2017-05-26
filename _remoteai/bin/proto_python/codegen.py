"""Runs protoc with the gRPC plugin to generate messages and gRPC stubs."""

from grpc.tools import protoc

protoc.main(
    (
	'',
	'-I../proto',
	'--python_out=.',
	'--grpc_python_out=.',
	'../proto/common.proto',
    '../proto/train.proto',
    )
)
